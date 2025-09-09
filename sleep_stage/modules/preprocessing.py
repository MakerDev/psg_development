import numpy as np
from .signals.filters import band_pass_filter

def prep_psg_signal(x, transpose=True, fs=50, hf=24.999, normalize=True): # 24.999 original

    eeg = np.array([
        band_pass_filter(x=x[i],lf=0.5, hf=hf, fs=fs)
        for i in range(6)
    ])  # (ch, tick) -> (6, 1438500)

    eog = np.array([
        band_pass_filter(x=x[i],lf=0.5, hf=hf, fs=fs)
        for i in range(6,8)
    ])  # (ch, tick) -> (2, 1438500)

    emg = band_pass_filter(x=x[-1],lf=10, hf=99, fs=fs)[np.newaxis,:]
        # (ch, tick) -> (1, 1438500)

    x = np.vstack([eeg,eog,emg])
        # (ch, tick) -> (9, 1438500)

    del eeg, eog, emg
        # memory free 

    if not normalize:
        center = np.mean(x, axis=1) # (9,)
        scale = np.std(x, axis=1)   # (9,)
        scale[scale == 1] = 1.0

        x = (x.T - center) / scale
        # (tick, ch) -> (1438500, 9) Transposed

    if not transpose: x = x.T

    return x


def prep_psg_signal_with_missing(
    x,                # shape: (len_x, ticks)
    transpose=True, 
    fs=50, 
    missing_channels=[]
):
    # 1) 남은 채널 목록
    channels_present = [ch for ch in range(9) if ch not in missing_channels]
    # sanity check
    if len(channels_present) != x.shape[0]:
        raise ValueError(
            f"Mismatch: x has shape[0]={x.shape[0]} but channels_present has len={len(channels_present)}"
        )
    
    # 2) band-pass filter 각 채널
    filtered_list = []
    lf, hf = 0.5, 24.999
    for i, ch_idx in enumerate(channels_present):
        # x[i] => ch_idx번 채널의 신호
        filtered = band_pass_filter(x=x[i], lf=lf, hf=hf, fs=fs)
        filtered_list.append(filtered[np.newaxis, :])  # shape => (1, ticks)

    # 3) stack => shape (len_x, ticks)
    x_filtered = np.vstack(filtered_list)

    # 4) standardize (channel-wise)
    center = np.mean(x_filtered, axis=1)  # shape=(len_x,)
    scale  = np.std(x_filtered, axis=1)   # shape=(len_x,)
    scale[scale==0] = 1.0

    x_filtered = (x_filtered.T - center) / scale

    if not transpose:
        x_filtered = x_filtered.T  # shape => (len_x, ticks)

    return x_filtered