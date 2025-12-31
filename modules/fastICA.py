from sklearn.decomposition import FastICA
from scipy.signal import correlate, find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_fastica_all_channels(S3):
    ica = FastICA(n_components=S3.shape[1], random_state=0, whiten='unit-variance')
    S_ica = ica.fit_transform(S3)
    return S_ica

def qrs_detector_simple_fetal(sig, fs):
    win = int(0.08 * fs)
    template = np.hstack([np.linspace(-1,1,win//2), np.linspace(1,-1,win//2)])
    corr = correlate(sig, template, mode='same')
    corr = corr / (np.std(corr)+1e-6)
    distance = int(0.25*fs)
    height = np.mean(corr) + 1*np.std(corr)
    peaks,_ = find_peaks(corr, distance=distance, height=height)
    return peaks

def analyze_fhr_reliability(qrs, fs):
    if len(qrs) < 2:
        return None, None
    rr = np.diff(qrs) / fs
    fhr = 60 / rr
    med = np.median(fhr)
    outliers = np.sum(np.abs(fhr-med)>10)
    rel = 1 - outliers/len(fhr)
    return np.mean(fhr), rel

def evaluate_against_true(qrs_pred, qrs_true, fs):
    if len(qrs_pred)==0:
        return None, None, None
    diffs = []
    for q in qrs_pred:
        diffs.append(np.min(np.abs(qrs_true - q)))
    diffs = np.array(diffs)
    mae = np.mean(diffs)*1000/fs
    rmse = np.sqrt(np.mean(diffs**2))*1000/fs
    rel_dist = np.mean(diffs < (30/1000 * fs))
    return mae, rmse, rel_dist

def process_record_ICA(S3, true_fqrs_up, fs):
    S_ica = run_fastica_all_channels(S3)

    best_rel = -1
    best_idx = None
    best_fhr = None
    best_qrs = None

    for i in range(S_ica.shape[1]):
        comp = S_ica[:,i]
        qrs = qrs_detector_simple_fetal(comp, fs)

        if len(qrs)<3: 
            continue

        mean_fhr, rel = analyze_fhr_reliability(qrs, fs)
        if rel is None: 
            continue
        
        if 110 < mean_fhr < 190:  
            if rel > best_rel:
                best_rel = rel
                best_idx = i
                best_fhr = mean_fhr
                best_qrs = qrs

    if best_idx is None:
        return {
            "ica_best_comp": None,
            "ica_qrs": None,
            "ica_mean_fhr": None,
            "ica_rel": None,
            "ica_mae": None,
            "ica_rmse": None,
            "ica_rel_dist": None
        }

    mae, rmse, rel_dist = evaluate_against_true(best_qrs, true_fqrs_up, fs)

    return {
        "ica_best_comp": best_idx,
        "ica_qrs": len(best_qrs),
        "ica_mean_fhr": best_fhr,
        "ica_rel": best_rel,
        "ica_mae": mae,
        "ica_rmse": rmse,
        "ica_rel_dist": rel_dist
    }


def mean_metrics_ica(df):
    metrics_for_mean = ['ica_rel','reliability', 'mean_bpm_f','mean_real_bpm_f', 'sir_gain_dB', 'fSNR_dB','SIR','SNR', 'SE', 'PPV', 'F1', 'ACC', 'TP', 'FP', 'FN']
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


def detect_fetal_component_ICA(S_ica, fs, qrs_detector):
    """
    Analyze all ICA components and:
    - discard those that are too slow (MECG < 110 bpm)
    - choose the one with the most plausible FHR (120–170 bpm)
    - calculate reliability and mean FHR
    """
    best_idx = None
    best_rel = -1
    best_fhr = None
    best_qrs = None

    for i in range(S_ica.shape[1]):
        comp = S_ica[:, i]
        qrs = qrs_detector(comp, fs)

        if len(qrs) < 3:
            continue

        rr = np.diff(qrs) / fs
        fhr = 60 / rr
        mean_fhr = np.mean(fhr)

        # Heuristic: skip maternal range (40–110 bpm)
        if mean_fhr < 110 or mean_fhr > 190:
            continue

        # reliability
        med = np.median(fhr)
        outliers = np.sum(np.abs(fhr - med) > 10)
        rel = 1 - outliers / len(fhr)

        if rel > best_rel:
            best_rel = rel
            best_idx = i
            best_fhr = fhr
            best_qrs = qrs

    return best_idx, best_qrs, best_fhr, best_rel


def compare_martens_vs_ica(
    qrs_martens,       # martens QRS
    qrs_true,          # real QRS
    S_ica,             # ICA comp 
    fs_up,             
    qrs_detector, 
    HR,
    reliability        
):
    # ICA: search fetal
    best_idx, qrs_ica, fhr_ica, rel_ica = detect_fetal_component_ICA(S_ica, fs_up, qrs_detector)
    
    print("=== ICA ===")
    print("Best fetal component:", best_idx)
    print("QRS ICA:", len(qrs_ica) if qrs_ica is not None else 0)
    print("Mean FHR ICA:", np.mean(fhr_ica) if fhr_ica is not None else None)
    print("Reliability ICA:", rel_ica)

 
    print("\n=== PIPELINE ===")
    print("QRS Pipeline:", len(qrs_martens))
    print("Mean FHR Pipeline:", np.mean(HR))
    print("Reliability Pipeline:", reliability)

    # QRS error vs real
    mae_ica, rmse_ica, rel_qrs_ica = evaluate_against_true(qrs_ica, qrs_true, fs_up)
    mae_m,  rmse_m,  rel_qrs_m  = evaluate_against_true(qrs_martens, qrs_true, fs_up)

    print("\n=== QRS ERROR VS REAL ===")
    print(f"ICA     → MAE: {mae_ica:.2f} ms | RMSE: {rmse_ica:.2f} ms | Reliability(dist): {rel_qrs_ica:.2f}")
    print(f"PIPELINE → MAE: {mae_m:.2f} ms | RMSE: {rmse_m:.2f} ms | Reliability(dist): {rel_qrs_m:.2f}")

    # dictionary 
    return {
        "ICA": {
            "component": best_idx,
            "QRS": len(qrs_ica) if qrs_ica is not None else 0,
            "FHR": np.mean(fhr_ica) if fhr_ica is not None else None,
            "RelFHR": rel_ica,
            "MAE": mae_ica,
            "RMSE": rmse_ica,
            "RelQRS": rel_qrs_ica
        },
        "Pipeline": {
            "QRS": len(qrs_martens),
            "FHR": np.mean(HR),
            "RelFHR": reliability,
            "MAE": mae_m,
            "RMSE": rmse_m,
            "RelQRS": rel_qrs_m
        }
    }


def plot_reliability_means(df):
    mean_m = df["reliability"].mean()
    mean_i = df["ica_rel"].mean()

    plt.figure(figsize=(6,4))
    plt.bar(["Pipeline", "ICA"], [mean_m, mean_i],
            color=["royalblue", "coral"])

    plt.ylabel("Mean Reliability")
    plt.title("Mean Reliability: Pipeline vs ICA")
    plt.ylim(0,1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.show()

    print("Mean Pipeline reliability:", mean_m)
    print("Mean ICA reliability:", mean_i)


def plot_reliability_comparison(df):
    records = df["record"]
    x = np.arange(len(records))

    plt.figure(figsize=(14,5))

    plt.bar(x - 0.2, df["reliability"], width=0.4, label="Pipeline", color="royalblue")
    plt.bar(x + 0.2, df["ica_rel"], width=0.4, label="ICA", color="coral")

    plt.xticks(x, records, rotation=90)
    plt.ylabel("Reliability")
    plt.title("Reliability Comparison (Pipeline vs ICA)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()