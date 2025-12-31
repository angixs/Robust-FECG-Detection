import numpy as np

def compute_snr_sir_martens(S4, S5, S6, fQRS, fs, Nav=150):
 
    W = S6.shape[0]              
    half_W = W // 2
    n_ch = S5.shape[1]

    # Power of fetal ECG estimate (PF)
    PF = np.mean(S6**2, axis=0)  # shape (C,)

    # Lists for noise and interference power
    PN_list = [[] for _ in range(n_ch)]
    PM_list = [[] for _ in range(n_ch)]

    for q in fQRS[:Nav]:   
        st = q - half_W
        en = q + half_W

        if st < 0 or en >= len(S5):
            continue

        # Window from S5 
        S5_win = S5[st:en, :]  

        # Noise estimate: S5_win - S6
        noise_win = S5_win - S6   

        # MECG residual estimate: S4 - S5 (full signal)
        mecg_residual_win = (S4[st:en, :] - S5[st:en, :])

        for ch in range(n_ch):
            PN_list[ch].append(np.mean(noise_win[:, ch]**2))
            PM_list[ch].append(np.mean(mecg_residual_win[:, ch]**2))

    # Convert to arrays
    PN = np.array([np.mean(p) if len(p) > 0 else 1e-12 for p in PN_list])
    PM = np.array([np.mean(p) if len(p) > 0 else 1e-12 for p in PM_list])

    # Compute SNR and SIR
    SNR = PF / PN
    SIR = PF / PM

    SNR_dB = 10 * np.log10(SNR + 1e-12)
    SIR_dB = 10 * np.log10(SIR + 1e-12)
    SNR=np.mean(SNR_dB)
    SIR=np.mean(SIR_dB)
    
    return SNR, SIR

def calculate_paper_metrics(f_peaks, fs, total_samples):

    if len(f_peaks) < 5:
        return 0.0, 0.0, False

    rr_sec = np.diff(f_peaks) / fs
    fhr_trace = 60.0 / rr_sec
    
    fhr_times = f_peaks[1:] / fs
    
    duration_sec = total_samples / fs
    
    n_outliers = 0
    total_points = len(fhr_trace)
    
    for t_start in range(0, int(duration_sec), 10):
        t_end = t_start + 10
        
        mask = (fhr_times >= t_start) & (fhr_times < t_end)
        block_fhr = fhr_trace[mask]
        
        if len(block_fhr) == 0:
            continue
            
        block_median = np.median(block_fhr)
        
        outliers_in_block = np.sum(np.abs(block_fhr - block_median) > 10)
        n_outliers += outliers_in_block

    reliability = 1.0 - (n_outliers / total_points)
    
    mean_bpm = np.mean(fhr_trace)
    

    is_successful = (60 <= mean_bpm <= 220) and (reliability > 0.60)
    
    return reliability, mean_bpm, is_successful


def compute_metrics(det, true, fs):
    tol = int(0.05 * fs)  # 50 ms

    true = np.array(true, dtype=int)
    det  = np.array(det, dtype=int)

    used = np.zeros(len(true), dtype=bool)

    TP = 0
    FP = 0

    # ---- MATCHING 1-to-1 (rigoroso) ----
    for d in det:
        dif = np.abs(true - d)
        idx = np.argmin(dif)
        if dif[idx] <= tol and not used[idx]:
            TP += 1
            used[idx] = True
        else:
            FP += 1

    FN = np.sum(~used)

    SE  = TP / (TP + FN + 1e-9)
    PPV = TP / (TP + FP + 1e-9)
    F1  = 2 * SE * PPV / (SE + PPV + 1e-9)

    ACC = TP / (TP + FP + FN)
     

    return {
        "SE": round(SE,4),
        "PPV": round(PPV,4),
        "F1": round(F1,4),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "ACC": round(ACC,4),
    }

