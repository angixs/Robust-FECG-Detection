import numpy as np


def mecg_canceller(S4, mQRS, fs, N=10):
    """    
    S4 : array (Nsamples × channels)
    mQRS : maternal QRS indices (array)
    fs : frequency (2000 Hz)
    N : number of previous templates (default = 10)

    Returns:
    S5 : signal after MECG cancellation
    """ 

    S = S4.copy().astype(float)
    n_samples, n_ch = S.shape

    # Window definitions 
    # μ(t) is split into three subtemplates μP, μQRS and μT, each scaled independently before subtraction.
    # P Wave: - 250ms to - 50ms
    P_before = int(0.25 * fs)    # 500
    P_after  = int(0.05 * fs)    # 100
    Np = P_before - P_after      # 400 samples

    # QRS: -50ms to +50ms
    QRS_half = int(0.05 * fs)    # 100
    Nq = 2 * QRS_half            # 200 samples

    # T Wave: +50ms to +450ms    
    T_before = int(0.05 * fs)    # 100
    T_after  = int(0.45 * fs)    # 900
    Nt = T_after - T_before      # 800 samples

    total_len = Np + Nq + Nt     # 400 + 200 + 800 = 1400

    for ch in range(n_ch):

        for idx, R in enumerate(mQRS):

            st = R - P_before
            en = R + T_after

            if st < 0 or en > n_samples:
                continue

            # template averaging
            prev_R = mQRS[max(0, idx - N):idx]

            if len(prev_R) < 3:
                continue  

            seg_list = []
            for Rp in prev_R:
                stp = Rp - P_before
                enp = Rp + T_after
                if stp >= 0 and enp <= n_samples:
                    seg_list.append(S[stp:enp, ch])

            if len(seg_list) == 0:
                continue

            mu = np.mean(seg_list, axis=0)
            m  = S[st:en, ch].copy()

            # enery normalization
            mu_std = np.std(mu) + 1e-8
            m_std  = np.std(m)  + 1e-8

            mu = mu / mu_std
            m  = m  / m_std


            # splitting mu in three parts
            mu_P   = mu[0:Np]
            mu_QRS = mu[Np:Np+Nq]
            mu_T   = mu[Np+Nq:Np+Nq+Nt]

            # smoothing 
            K = 5 

            # P → QRS
            mu_P[-K:]   = 0.5*mu_P[-K:]   + 0.5*mu_QRS[:K]
            mu_QRS[:K]  = mu_P[-K:]       # match continuity

            # QRS → T
            mu_QRS[-K:] = 0.5*mu_QRS[-K:] + 0.5*mu_T[:K]
            mu_T[:K]    = mu_QRS[-K:]

            # matrix
            M = np.zeros((total_len, 3))
            M[0:Np, 0]           = mu_P
            M[Np:Np+Nq, 1]       = mu_QRS
            M[Np+Nq:Np+Nq+Nt, 2] = mu_T

            # aP, aQRS, aT
            a = np.linalg.pinv(M) @ m    
            m_hat_norm = M @ a
            m_hat = m_hat_norm * m_std  
     
            S[st:en, ch] -= m_hat

    return S

