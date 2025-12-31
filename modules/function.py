import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz
cmap = plt.cm.plasma
colors = [cmap(0.8), cmap(0.3),cmap(0.4),cmap(0.5)]  

def plot_fir_response(fs, fc=3, numtaps=1001):
    # FIR coefficients
    b = firwin(numtaps, fc/(fs/2), pass_zero=False)

    # Frequency response
    w, h = freqz(b, worN=4096)

    # Convert to Hz
    f = w * fs / (2*np.pi)

    # Magnitude
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(f, 20*np.log10(np.abs(h) + 1e-12))
    plt.axvline(fc, color='r', linestyle='--', label='Cutoff')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Magnitude Response')
    plt.grid(True)
    plt.legend()

    # Phase
    plt.subplot(1,2,2)
    plt.plot(f, np.unwrap(np.angle(h)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (rad)')
    plt.title('Phase Response')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_sampling_comparison(X, X_up, fs, fs_up, ch=0, t_ms=10):
    N  = int((t_ms/1000) * fs)
    Nu = int((t_ms/1000) * fs_up)

    t  = np.arange(N) / fs * 1000
    tu = np.arange(Nu) / fs_up * 1000

    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.stem(t, X[:N, ch], linefmt=colors[0], markerfmt=colors[0], basefmt=" ")
    plt.title(f"{fs} Hz Sampling")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.stem(tu, X_up[:Nu, ch], linefmt=colors[1], markerfmt=colors[1], basefmt=" ")
    plt.title(f"{fs_up} Hz Sampling")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def c_mean_bpm(true_qrs_indices, fs):
    RR_samples = np.diff(true_qrs_indices)
    
    if len(RR_samples) < 1:
        return np.nan 

    RR_seconds = RR_samples / fs
    
    HR_bpm = 60.0 / RR_seconds
    
    mean_bpm = np.mean(HR_bpm)
    median_bpm = np.median(HR_bpm)
    
    return median_bpm

def calculate_snr_sir(s4, s5, s6):
 
    eps = 1e-10
    def power(x):
        return np.mean(x**2, axis=0)
    
    P_F = power(s6)
    mecg_est = s4 - s5
    P_M = power(mecg_est)
    noise_est = s5 - s6
    P_N = power(noise_est)
    
    snr_linear = P_F / (P_N + eps)
    sir_linear = P_F / (P_M + eps)
    
    snr_db = 10 * np.log10(snr_linear)
    sir_db = 10 * np.log10(sir_linear)
    
    return np.mean(snr_db), np.mean(sir_db)

def compute_FHR(fQRS, fs_up):
    fQRS = np.array(fQRS)
    RR = np.diff(fQRS) / fs_up  # sec
    FHR = 60 / RR               # bpm
    return FHR


def mean_metrics(df):
    metrics_for_mean = ['reliability', 'mean_bpm_f','mean_real_bpm_f', 'sir_gain_dB', 'fSNR_dB','SIR','SNR', 'SE', 'PPV', 'F1', 'ACC', 'TP', 'FP', 'FN']
    df_mean = df[metrics_for_mean].mean().round(4).rename('Global Mean')
    df_std = df[metrics_for_mean].std().round(4).rename('Global Dev. Std')

 
    df_global_summary = pd.concat([df_mean, df_std], axis=1)

    print("\n## 1. Global Performance(Mean and Dev. Std)")
    print(df_global_summary)

    print("\n--- Analysis 2: Group using isSuccess ---")

    df_grouped = df.groupby('isSuccess')[metrics_for_mean].mean().round(4)

    print("\n## 2. Comparison between SUCCESS & FAIL")

    df_grouped.index = ['FAIL (False)', 'SUCCESS (True)']
    print(df_grouped)

    print("\n--- Analysis 3: Differences between groups ---")

    if 'SUCCESS (True)' in df_grouped.index and 'FAIL (False)' in df_grouped.index:
        df_diff = (df_grouped.loc['SUCCESS (True)'] - df_grouped.loc['FAIL (False)']).round(4).rename('Differences (SUCCESS - FAIL)')
        print("\n## 3. Difference mean (Success vs. Fail)")
        print(df_diff)
    else:
        print("Error")


def calculate_average_fecg_from_pc(signal_pc, fqrs_indices, fs_up, win_ms=100, num_avg=150):
   
    win = int((win_ms / 1000) * fs_up)
    
    segments = []
    
    qrs_to_use = fqrs_indices[:num_avg] 
    for q in qrs_to_use: 
        if q - win >= 0 and q + win < len(signal_pc):
            seg = signal_pc[q - win : q + win]
            segments.append(seg)
    
    Nav = len(segments) 
    if segments:
        S6_pc = np.mean(segments, axis=0)
    else:
        S6_pc = np.zeros(2 * win)

    return S6_pc, Nav

def plot_reliability_all(df):
    colors = ["mediumseagreen" if s else "lightcoral" for s in df["isSuccess"]]

    plt.figure(figsize=(12,5))
    plt.bar(df["record"], df["reliability"], color=colors)

    plt.xticks(rotation=90)
    plt.ylabel("Reliability")
    plt.title("Reliability Record")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_all_metrics_groups(df, metrics):
    """
    - FAIL (False)
    - SUCCESS (True)
    - GLOBAL MEAN
    """


    fail_means = df[df["isSuccess"] == False][metrics].mean()
    success_means = df[df["isSuccess"] == True][metrics].mean()
    global_means = df[metrics].mean()

    fail_vals = fail_means.values
    success_vals = success_means.values
    global_vals = global_means.values

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10,6))

    rects1 = ax.bar(x - width, fail_vals, width, label="FAIL (False)", color="lightcoral")
    rects2 = ax.bar(x,        success_vals, width, label="SUCCESS (True)", color="mediumseagreen")
    rects3 = ax.bar(x + width, global_vals, width, label="GLOBAL MEAN", color="lightblue")

    ax.set_xticks(x, metrics)
    ax.set_ylabel("Value")
    ax.set_title("Comparison Across Groups")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.show()



def scatter_plot_F1(df):
    plt.figure(figsize=(8, 6))
 
    scatter = plt.scatter(df['fSNR_dB'], df['F1'], c=df['isSuccess'], cmap='RdYlGn', alpha=0.7) 

    plt.title('F1-Score vs. Fetal SNR (fSNR_dB)')
    plt.xlabel('Fetal SNR (dB)')
    plt.ylabel('F1-Score')
    plt.grid(True, linestyle='--', alpha=0.6)

    
    legend1 = plt.legend(*scatter.legend_elements(), 
                         title="Success", 
                         labels=['FAIL (False)', 'SUCCESS (True)'])
    plt.gca().add_artist(legend1)
    plt.show()


def plot_distribution(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df['F1'], bins=10, edgecolor='black', color='skyblue')
    plt.title('Distribution of F1-Score')
    plt.xlabel('F1-Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
 


def plot_success_vs_fail_metrics(df):

    metrics_for_plot = ['F1', 'SE', 'PPV', 'sir_gain_dB', 'fSNR_dB', 'TP', 'FP', 'FN']
    df_grouped = df.groupby('isSuccess')[metrics_for_plot].mean().round(4)
   
    fail_data = df_grouped.loc[False]
    success_data = df_grouped.loc[True]
 
    
    
    performance_metrics = ['F1', 'SE', 'PPV']
    

    noise_metrics = ['sir_gain_dB', 'fSNR_dB']
    
    error_metrics = ['FP', 'FN']

    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    width = 0.35
    x = np.arange(len(performance_metrics))

    
    ax = axes[0]
    rects1 = ax.bar(x - width/2, fail_data[performance_metrics], width, label='FAIL (False)', color='lightcoral')
    rects2 = ax.bar(x + width/2, success_data[performance_metrics], width, label='SUCCESS (True)', color='mediumseagreen')

    ax.set_ylabel('Score (0-1)')
    ax.set_title('Performance')
    ax.set_xticks(x, performance_metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    

    ax = axes[1]
    noise_x = np.arange(len(noise_metrics))
    rects3 = ax.bar(noise_x - width/2, fail_data[noise_metrics], width, label='FAIL (False)', color='lightcoral')
    rects4 = ax.bar(noise_x + width/2, success_data[noise_metrics], width, label='SUCCESS (True)', color='mediumseagreen')

    ax.set_ylabel('Value (dB)')
    ax.set_title('SIR Gain e Fetal SNR')
    ax.set_xticks(noise_x, noise_metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    autolabel(rects3, ax)
    autolabel(rects4, ax)
    
  
    ax = axes[2]
    error_x = np.arange(len(error_metrics))
    rects5 = ax.bar(error_x - width/2, fail_data[error_metrics], width, label='FAIL (False)', color='lightcoral')
    rects6 = ax.bar(error_x + width/2, success_data[error_metrics], width, label='SUCCESS (True)', color='mediumseagreen')

    ax.set_ylabel('Count')
    ax.set_title('Comparison Error')
    ax.set_xticks(error_x, error_metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    autolabel(rects5, ax)
    autolabel(rects6, ax)
    
    plt.suptitle("Statistical Analysis:Comparison Mean Performance for Success", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Regola layout per titolo principale
    plt.show()

def plot_reliability_vs_snr_sir(df):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,5))

  
    ax1 = plt.subplot(1,2,1)
    x = df["SNR"].values
    y = df["reliability"].values * 100

    ax1.scatter(x, y, c="mediumseagreen", edgecolor="black")


    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(np.unique(x_clean)) >= 2:
        coeff = np.polyfit(x_clean, y_clean, deg=1)
        poly = np.poly1d(coeff)
        x_line = np.linspace(min(x_clean), max(x_clean), 200)
        ax1.plot(x_line, poly(x_line), "--", color="black", linewidth=2)
    else:
        print("imp")

    ax1.set_xlabel("SNR [dB]")
    ax1.set_ylabel("Reliability [%]")
    ax1.set_title("Reliability vs SNR")
    ax1.grid(alpha=0.4)


    ax2 = plt.subplot(1,2,2)
    x = df["SIR"].values
    y = df["reliability"].values * 100

    ax2.scatter(x, y, c="royalblue", edgecolor="black")

    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(np.unique(x_clean)) >= 2:
        coeff = np.polyfit(x_clean, y_clean, deg=1)
        poly = np.poly1d(coeff)
        x_line = np.linspace(min(x_clean), max(x_clean), 200)
        ax2.plot(x_line, poly(x_line), "--", color="black", linewidth=2)
    else:
        print("imp.")

    ax2.set_xlabel("SIR [dB]")
    ax2.set_ylabel("Reliability [%]")
    ax2.set_title("Reliability vs SIR")
    ax2.grid(alpha=0.4)

    plt.tight_layout()
    plt.show()



#for ICA comparison

def plot_success_fail_means(df):
    df_succ = df[df["isSuccess"] == True]
    df_fail = df[df["isSuccess"] == False]

    metrics = ["reliability", "ica_rel"]

    labels = ["Pipeline Reliability", "ICA Reliability"]

    mean_succ = df_succ[metrics].mean().values
    mean_fail = df_fail[metrics].mean().values

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, mean_fail, width, label="FAIL", color="lightcoral")
    plt.bar(x + width/2, mean_succ, width, label="SUCCESS", color="mediumseagreen")

    plt.xticks(x, labels)
    plt.ylabel("Mean Value")
    plt.title("Success vs Fail — Pipeline vs ICA")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.show()

    print("SUCCESS means:\n", df_succ[metrics].mean())
    print("\nFAIL means:\n", df_fail[metrics].mean())
