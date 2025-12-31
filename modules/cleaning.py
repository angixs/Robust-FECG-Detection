import numpy as np
from scipy.signal import firwin, filtfilt
from scipy.signal import resample_poly

def baseline_wander_removal(X, fs, fc=3, numtaps=1001): 
    b = firwin(numtaps, fc/(fs/2), pass_zero=False)
    return filtfilt(b, [1.0], X, axis=0)

def powerline_interference_canceller(
        S2, fs,
        f0=50,
        harmonics=[1,2,3],
        mu_w=1e-4,      
        mu_f=1e-7,        
    ):
    """
    - Estimates amplitude, phase, and frequency
    - Sine + cosine model for each channel
    - Adaptive frequency update
    - Multiharmonic approach
    """

    S2 = np.nan_to_num(S2)
    N, ch = S2.shape
    S3 = S2.copy()

    t = np.arange(N) / fs

    for h in harmonics:
        f = f0 * h
        omega = 2 * np.pi * f

        # coeff. per channel
        w1 = np.zeros(ch)  # sin component
        w2 = np.zeros(ch)  # cos component

        S_temp = S3.copy()

        phi = 0.0  # initial phase

        for n in range(N):

            # adaptive sinusoid + cosinusoid
            r1 = np.sin(phi)
            r2 = np.cos(phi)

            # prediction
            y = w1 * r1 + w2 * r2
            e = S_temp[n] - y

            # weight adaptive - LMS (multi-channel)
            w1 += 2 * mu_w * e * r1
            w2 += 2 * mu_w * e * r2

            y_prime = w1 * r2 - w2 * r1    

            # update phase:
            phi += (omega + mu_f * np.sum(e * y_prime))

            if phi > 2*np.pi:
                phi -= 2*np.pi
            elif phi < 0:
                phi += 2*np.pi

            S_temp[n] = e

        S3 = S_temp

    return S3

def upsample_to_2kHz(X, fs, target_fs=2000):
    L = target_fs // fs       
    M = 1                     
    X_up = resample_poly(X, L, M, axis=0, window=('kaiser', 5.0))
    return X_up, target_fs