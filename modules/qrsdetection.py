import numpy as np
from scipy.signal import correlate

# Maternal QRS Detection
def pca_first_component(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(X, axis=0)
    good = std > 1e-8
    X = X[:, good]

    if X.shape[1] == 1:
        return X[:, 0]

    Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    C = np.dot(Xn.T, Xn) / Xn.shape[0]
    eigvals, eigvecs = np.linalg.eigh(C)
    v1 = eigvecs[:, np.argmax(eigvals)]

    PC1 = Xn @ v1
    return PC1

# first fix template 
def build_initial_qrs_template(fs):
    t = np.linspace(-0.05, 0.05, int(0.1*fs))
    template = -np.exp(-(t**2)/(2*(0.01**2))) + 0.5*np.exp(-(t**2)/(2*(0.015**2)))
    return template / np.max(np.abs(template))

def adaptive_threshold(corr, window=5000, perc=98):
    thr = np.zeros_like(corr)
    half = window // 2

    for i in range(len(corr)):
        start = max(0, i - half)
        end   = min(len(corr), i + half)
        thr[i] = np.percentile(corr[start:end], perc)

    return thr

def extract_qrs_from_corr(corr, fs, perc=98, refr_ms=300, refine_ms=40):
    thr = adaptive_threshold(corr, perc=perc)
    cand = np.where(corr > thr)[0]

    qrs = []
    last = -1_000_000
    refr = int((refr_ms/1000) * fs)
    w    = int((refine_ms/1000) * fs)

    for c in cand:
        if c - last < refr:
            continue

        st = max(0, c - w)
        en = min(len(corr), c + w)

        peak = st + np.argmax(corr[st:en])
        qrs.append(peak)
        last = peak

    return np.array(qrs)

# real template based on some QRS detected
def build_real_qrs_template(signal, peaks, fs, pre_ms=50, post_ms=50):
    w_pre  = int((pre_ms/1000) * fs)
    w_post = int((post_ms/1000) * fs)

    segments = []

    for R in peaks:
        if R - w_pre < 0 or R + w_post > len(signal):
            continue
        seg = signal[R - w_pre : R + w_post]
        segments.append(seg)

    if len(segments) == 0:
        raise ValueError("No valid QRS segments for building real template")

    segments = np.array(segments)
    template = np.mean(segments, axis=0)
    template = template / np.max(np.abs(template))  # normalize amplitude

    return template

def detect_mQRS(S4, fs):
    # 1) PCA 
    PC1 = pca_first_component(S4)

    # 2) Rough detection with synthetic template 
    template_init = build_initial_qrs_template(fs)

    corr1 = correlate(PC1, template_init, mode='same')
    rough_peaks = extract_qrs_from_corr(corr1, fs, perc=98)

    # 3) Build real template from rough peaks 
    template_real = build_real_qrs_template(PC1, rough_peaks, fs)

    # 4) Final correlation with REAL template 
    corr2 = correlate(PC1, template_real, mode='same')
    final_peaks = extract_qrs_from_corr(corr2, fs, perc=98)

    return final_peaks, PC1, corr2, template_real


# Fetal QRS Detection
def build_initial_fetal_template(fs):
    t = np.linspace(-0.03, 0.03, int(0.06 * fs))
    template = -np.exp(-(t**2)/(2*(0.005**2))) + 0.4*np.exp(-(t**2)/(2*(0.008**2)))
    return template / np.max(np.abs(template))

def build_real_fetal_template(signal, peaks, fs, pre_ms=20, post_ms=20):
    w_pre = int((pre_ms/1000) * fs)
    w_post = int((post_ms/1000) * fs)

    segments = []
    for R in peaks:
        if R - w_pre < 0 or R + w_post > len(signal):
            continue
        seg = signal[R - w_pre : R + w_post]
        segments.append(seg)

    if len(segments) == 0:
        raise ValueError("No valid fetal QRS segments")

    segments = np.array(segments)
    template = np.mean(segments, axis=0)
    template = template / np.max(np.abs(template))
    return template

def detect_fQRS(S5, fs):

    # 1) PCA enhancement 
    PC1 = pca_first_component(S5)

    # 2) Rough detection with synthetic fetal template 
    T_init = build_initial_fetal_template(fs)
    corr1 = correlate(PC1, T_init, mode='same')
    rough_peaks = extract_qrs_from_corr(corr1, fs, perc=98, refr_ms=150, refine_ms=20)

    # 3) Build REAL fetal template 
    T_real = build_real_fetal_template(PC1, rough_peaks, fs)

    # 4) Final detection with real template 
    corr2 = correlate(PC1, T_real, mode='same')
    fQRS = extract_qrs_from_corr(corr2, fs, perc=98, refr_ms=150, refine_ms=20)

    return np.array(fQRS), PC1, corr2, T_real
