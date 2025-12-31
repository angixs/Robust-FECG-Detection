import matplotlib.pyplot as plt
import numpy as np

def FECG_avg(S5, clean_fQRS, fs_up):
    win_ms = 100
    win = int((win_ms/1000) * fs_up)

    S6 = np.zeros((2*win, S5.shape[1]))

    for ch in range(S5.shape[1]):
        segments = []
        for q in clean_fQRS[:150]:  
            if q-win >= 0 and q+win < len(S5):
                seg = S5[q-win:q+win, ch]
                segments.append(seg)
        S6[:, ch] = np.mean(segments, axis=0)

    plt.figure(figsize=(9,4))
    for i in range(S6.shape[1]):
        plt.plot(S6[:, i], color=plt.cm.plasma(i / S6.shape[1]))
        plt.legend([f"AECG {i+1}" for i in range(S6.shape[1])])
    plt.title("Average fetal ECG complex")
    plt.show()
    return S6