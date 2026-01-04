import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.fft import fft, fftfreq


cmap = plt.cm.plasma
colors = [cmap(0.8), cmap(0.3),cmap(0.4),cmap(0.5), cmap(0.2), cmap(0.1)]  

# Calculate Metrics
def calculate_sir_gain(S_before, S_after, mQRS, fs):
    """
    Estimate the SIR gain (mECG interference reduction) 
    by measuring the reduction in synchronized signal power on mQRS.
    """
    window_ms = 100 # ±50 ms
    half_window = int(0.05 * fs) 
    
    mECG_segments_before = []
    mECG_segments_after = []
    
    N_ch = S_before.shape[1]
    
    for R in mQRS:
        st = R - half_window
        en = R + half_window
        
        if st >= 0 and en <= len(S_before):
            mECG_segments_before.append(np.mean(S_before[st:en, :]**2))
            mECG_segments_after.append(np.mean(S_after[st:en, :]**2))
            
    if not mECG_segments_before:
        return 0

    Power_I_before = np.mean(mECG_segments_before)
    Power_I_after = np.mean(mECG_segments_after)
    
    # SIR Gain in dB
    SIR_Gain_dB = 10 * np.log10(Power_I_before / Power_I_after)
    
    return SIR_Gain_dB

def calculate_fetal_snr(S5, fQRS, fs):
    """
    Estimate the fetal SNR by comparing the energy of the fQRS with the energy of the noise 
    in the intervals between peaks.
    """
    # Define windows
    qrs_window_ms = 100  
    noise_window_ms = 200 
    
    qrs_half = int(0.05 * fs)
    noise_half = int(0.1 * fs) 
    
    fECG_power = []
    Noise_power = []

    for i in range(len(fQRS) - 1):
        R_curr = fQRS[i]
        R_next = fQRS[i+1]
        
        # Power fECG 
        st_sig = R_curr - qrs_half
        en_sig = R_curr + qrs_half
        if st_sig >= 0 and en_sig < len(S5):
            fECG_power.append(np.mean(S5[st_sig:en_sig, :]**2))

        # Power Noise 
        mid_point = (R_curr + R_next) // 2
        st_noise = mid_point - noise_half
        en_noise = mid_point + noise_half
        
 
        if st_noise > en_sig and en_noise < R_next - qrs_half and st_noise < en_noise:
            Noise_power.append(np.mean(S5[st_noise:en_noise, :]**2))
    
    if not fECG_power or not Noise_power:
        return 0

    Power_S = np.mean(fECG_power)
    Power_N = np.mean(Noise_power)

    fSNR_dB = 10 * np.log10(Power_S / Power_N)
    
    return fSNR_dB

def plot_before_after(X,S2,fs, string):
    start_sec = 10  
    end_sec = 15    
    start_sample = int(start_sec * fs)
    end_sample = int(end_sec * fs)

    ch = 0 
    time_vector = np.arange(start_sample, end_sample) / fs

    plt.figure(figsize=(12, 6))
    cmap = plt.cm.plasma

    plt.subplot(2, 1, 1)
    plt.plot(time_vector, X[start_sample:end_sample, ch], 
             label="Before", 
             color=colors[0], 
             linewidth=1)
    plt.title(f"Before {string} - AECG {ch+1}")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2) 
    plt.plot(time_vector, S2[start_sample:end_sample, ch], 
             label="After", 
             color=colors[1], 
             linewidth=1)
    plt.title(f"After {string} - AECG{ch+1}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot Function 
def initial_plot(X,S2):
    for ch in range(0, 4):
        plt.figure(figsize=(24,6))
        plt.plot(X[:,ch],color=colors[2], label='Raw')
        plt.plot(S2[:,ch], color=colors[0],label='After BW removal')
        plt.legend()
        plt.title(f"Step 1:AECG{ch+1}")
        plt.show()


def plot_power(S2,fs):
    f, Pxx = welch(S2[:,0], fs, nperseg=4096)

    plt.figure(figsize=(12,4))
    plt.axvline(50, color=colors[0], linestyle='--')
    plt.axvline(100, color=colors[0], linestyle='--')
    plt.axvline(150, color=colors[0], linestyle='--')
    plt.semilogy(f, Pxx, color=colors[1])
    plt.title("PSD – Powerline check")
    plt.xlabel("Hz")
    plt.ylabel("PSD")
    plt.grid(True)

    return plt.show()


def plot_check_zoom(S3,S4, fs_up, fs):
    i = 10000       
    w = 10        
    scale = fs_up / fs

    orig = S3[i-w:i+w, 0]
    start = int((i-w) * scale)
    end   = int((i+w) * scale)
    up = S4[start:end, 0]

    t_orig = np.linspace(0, 1, len(orig))
    t_up   = np.linspace(0, 1, len(up))

    plt.figure(figsize=(12,5))
    plt.plot(t_orig, orig, label='Original', color=colors[0])
    plt.plot(t_up, up, label='Upsampled', color=colors[1], alpha=0.7)
    plt.legend()
    plt.title("Upsampling - Correct Zoom Check")
    return plt.show()

def all_channel_shape(S4):
    c = 0
    plt.figure(figsize=(24,6))
    for ch in range(S4.shape[1]):
        plt.plot(S4[:5000,ch] + 50*ch, color=colors[c])
        c=c+1
    plt.title("All channels stacked")
    return plt.show()
 
def plot_template_m(maternal_template_clean):
    plt.figure(figsize=(8,4))
    plt.plot(maternal_template_clean, color=colors[1])
    plt.title("Maternal QRS Template")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    return plt.show()

def plot_hrf(fQRS, fs_up, HR, string):
    time_vector_seconds = fQRS[1:] / fs_up

    plt.figure(figsize=(16, 4))
    plt.plot(time_vector_seconds, HR, marker='o', color='orange')
    plt.axhline(np.mean(HR), label="Mean HR", color='red')
    plt.title(f"{string} Heart Rate")
    plt.xlabel("Time (sec)")
    plt.ylabel("BPM")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_qrs_det_m(PC1_maternal_clean,mQRS_clean):
    plt.figure(figsize=(18,5))
    plt.plot(PC1_maternal_clean, label="PC1 (Maternal)", linewidth=1.0, color=colors[1])

    plt.scatter(
        mQRS_clean,
        PC1_maternal_clean[mQRS_clean],
        color='red',
        s=40,
        label="Detected QRS"
    )

    plt.title("Maternal QRS Detection on PC1")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt.show()

def check_variance_reduction(S_original, S_cancelled):
    var_orig = np.var(S_original)
    var_canc = np.var(S_cancelled)
    reduction = (1 - var_canc / var_orig) * 100
    print(f"Total Variance Reduced: {reduction:.2f}%")

def plot_spectral_density_power(S4_pre,fs_up,S5):
    def plot_spectral_density(X, fs, label, color):
        N = len(X)
        yf = fft(X)
        xf = fftfreq(N, 1 / fs)
        idx = xf > 0
        plt.plot(xf[idx], 2.0/N * np.abs(yf[idx]), label=label, color=color)

    plt.figure()
    plot_spectral_density(S4_pre[:, 0], fs_up, 'S4 Before MECG', colors[0])
    plot_spectral_density(S5[:, 0], fs_up, 'S5 After MECG', colors[1])
    plt.legend()
    plt.title('Spectral Power')
    return plt.show()

def plot_reduction_on_ch(S4,S5):
    for ch in range(4):
        plt.figure(figsize=(15,4))
        plt.plot(S4[:,ch], color=colors[0],label='Original')
        plt.plot(S5[:,ch], color=colors[1], label='After MECG cancellation', )
        plt.title(f'AECG{ch+1}')
        plt.legend()
        plt.show()

def plot_single_after_MECG_canc(mQRS_clean, S4_pre, S5):
    i = mQRS_clean[8] 
    w = 800     

    plt.figure(figsize=(12,4))
    plt.plot(S4_pre[i-w:i+w,0],color=colors[0], label='Before',alpha=0.6    )
    plt.plot(S5[i-w:i+w,0],color=colors[1], label='After')
    plt.legend()
    plt.title("Zoom around maternal QRS after cancellation")
    plt.show()

def plot_ch_var(S4_pre, S5):
    var_S4 = np.var(S4_pre, axis=0)
    var_S5 = np.var(S5, axis=0)

    plt.figure(figsize=(10,4))
    plt.bar(np.arange(len(var_S4))-0.15, var_S4, width=0.3, label="Before MECG cancellation (Maternal-dominated)", color=colors[0])
    plt.bar(np.arange(len(var_S5))+0.15, var_S5, width=0.3, label="After MECG cancellation (Fetal-dominated)",color=colors[1])
    plt.xticks(np.arange(len(var_S4)), [f"AECG{i+1}" for i in range(len(var_S4))])
    plt.ylabel("Variance")
    plt.title("Channel variance: maternal vs fetal")
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

def plot_template_f(tmpl_f):
    plt.figure(figsize=(8,4))
    plt.plot(tmpl_f, color=colors[0])
    plt.title("Fetal QRS Template")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()


def plot_medium_S6(PC1_f, clean_fQRS, fs_up):
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

    S6_pc, Nav_used = calculate_average_fecg_from_pc(PC1_f, clean_fQRS, fs_up)
    peak_index = np.argmax(np.abs(S6_pc))

    if S6_pc[peak_index] < 0:
        S6_pc = -S6_pc

    plt.figure(figsize=(10,4))
    plt.plot(S6_pc, color=colors[3], linewidth=2)
    plt.title(f"Average Fetal ECG Complex, Nav={Nav_used})") 
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show() 

    return plt.show()

def plot_qrs_comparison(ecg_signal, true_qrs, detected_qrs, fs):
    
    ecg_signal = np.array(ecg_signal)
    true_qrs = np.array(true_qrs, dtype=int)
    detected_qrs = np.array(detected_qrs, dtype=int)


    time_axis = np.arange(len(ecg_signal)) / fs
   
    plt.figure(figsize=(24, 6))

    plt.plot(time_axis, ecg_signal, label='Signal ECG (Fetal)', color=colors[5], linewidth=1)

    plt.plot(
        time_axis[true_qrs], 
        ecg_signal[true_qrs], 
        'o', 
        label='Real QRS(Ground Truth)', 
        color='red', 
        markersize=6
    )

    plt.plot(
        time_axis[detected_qrs], 
        ecg_signal[detected_qrs], 
        'v', 
        label='Detected QRS', 
        color='green', 
        markersize=6
    )
    
    plt.title('Real Fetal vs Detect Fetal', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_qrs_zoom(ecg_signal, true_qrs, detected_qrs, fs, start_time_sec, duration_sec):
    
    ecg_signal = np.array(ecg_signal)
    true_qrs = np.array(true_qrs, dtype=int)
    detected_qrs = np.array(detected_qrs, dtype=int)

    time_axis = np.arange(len(ecg_signal)) / fs

    start_sample = int(start_time_sec * fs)
    end_sample = int((start_time_sec + duration_sec) * fs)

    start_sample = max(0, start_sample)
    end_sample = min(len(ecg_signal), end_sample)
    
    plt.figure(figsize=(24, 6))
    
    plt.plot(time_axis[start_sample:end_sample], 
             ecg_signal[start_sample:end_sample], 
             label='Signal ECG (Fetal)', 
             color=colors[5], 
             linewidth=1.5)

    true_qrs_zoom = true_qrs[(true_qrs >= start_sample) & (true_qrs < end_sample)]
    plt.plot(
        time_axis[true_qrs_zoom], 
        ecg_signal[true_qrs_zoom], 
        'o', 
        label='Real QRS(Ground Truth)', 
        color='red', 
        markersize=8,
        zorder=5
    )

    detected_qrs_zoom = detected_qrs[(detected_qrs >= start_sample) & (detected_qrs < end_sample)]
    plt.plot(
        time_axis[detected_qrs_zoom], 
        ecg_signal[detected_qrs_zoom], 
        'v', 
        label='Detect QRS', 
        color='green', 
        markersize=8,
        zorder=5
    )
    

    plt.xlim(start_time_sec, start_time_sec + duration_sec)

    plt.title(f'Zoom: [{start_time_sec}s - {start_time_sec + duration_sec}s]', fontsize=14)
    plt.xlabel('Time(s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def final_plot_comparison(X,fs_up,S3,S5,S6):
    channels = X.shape[1]
    fs = fs_up
    time_window = 2  
    L = int(fs * time_window)

    start = 0
    end = start + L

    S1_w = X[start:end, :]
    S3_w = S3[start:end, :]
    S5_w = S5[start:end, :]
    S6_w = S6[:, :] 


    fig, axs = plt.subplots(channels, 4, figsize=(14, 6), sharex=False, sharey=False)

    titles = ["S1 raw abdominal", "S3 cleaned",
              "S5 MECG cancelled", "S6 average fetal complex"]

    for c in range(channels):
        # Panel 1 – S1
        axs[c,0].plot(S1_w[:,c], color=colors[1], linewidth=0.8)
        axs[c,0].set_title(titles[0] if c==0 else "")
        axs[c,0].set_ylabel(f"AECG {c+1}")

        # Panel 2 – S3
        axs[c,1].plot(S3_w[:,c],color=colors[2], linewidth=0.8)
        axs[c,1].set_title(titles[1] if c==0 else "")

        # Panel 3 – S5
        axs[c,2].plot(S5_w[:,c], color=colors[3],linewidth=0.8)
        axs[c,2].set_title(titles[2] if c==0 else "")

        # Panel 4 – S6 (average FECG complex)
        axs[c,3].plot(S6_w[:,c], color=colors[0], linewidth=0.8)
        axs[c,3].set_title(titles[3] if c==0 else "")

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
