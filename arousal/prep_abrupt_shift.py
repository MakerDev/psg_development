import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import numpy as np
from os.path import basename, join 
from scipy.ndimage import uniform_filter1d
from utils.tools import *

channel_map = {
    "EEG-F3-M2": 0,
    "EEG-F4-M1": 1,
    "EEG-C3-M2": 2,
    "EEG-C4-M1": 3,
    "EEG-O1-M2": 4,
    "EEG-O2-M1": 5,
    "EEG-O2": 5,
}

def moving_window_mean_rms_norm(x, fs=50, window_min=18):
    window_size = int(window_min * 60 * fs)
    out = np.zeros_like(x, dtype=np.float32)

    for ch_idx in range(x.shape[0]):
        ch_data = x[ch_idx]

        mean_val = uniform_filter1d(ch_data, size=window_size, mode='reflect')

        sqr_val = ch_data**2
        rms_val = np.sqrt(
            uniform_filter1d(sqr_val, size=window_size, mode='reflect')
        )
        rms_val[rms_val < 1e-12] = 1e-12
        out[ch_idx] = (ch_data - mean_val) / rms_val

    return out

def create_arousal_labels_per_channel(events, meas_date, total_samples, sfreq=50):
    y = np.zeros((6, total_samples), dtype=np.float32)
    meas_date = meas_date.replace(tzinfo=None)

    for event in events:
        desc = event["description"]
        loc = event["location"]
        
        if desc == "MW_EEG-AS":
            if loc in channel_map:
                ch_idx = channel_map[loc]
            else:
                # 해당 location이 6채널 범위 아니면 무시
                print(f"Unknown location: {loc}")
                continue
            
            event_start = (event["onset"] - meas_date).total_seconds()
            event_end   = event_start + event["duration"]
            
            if event_start < 0:
                event_start = 0
            if event_end < 0:
                continue
            
            s_idx = int(event_start * sfreq)
            e_idx = int(event_end   * sfreq)
            
            if s_idx < 0:
                s_idx = 0
            if e_idx > total_samples:
                e_idx = total_samples
            
            if s_idx < total_samples:
                y[ch_idx, s_idx:e_idx] = 1.0
    return y


def pad_signals(x, y, max_len=2**22):
    x = np.transpose(x, (1, 0))
    curr_len = x.shape[1]
    padd = max_len - curr_len
    if padd > 0:
        left_pad = padd // 2 + padd % 2
        right_pad = padd // 2

        x = np.pad(x, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=0)
        y = np.pad(y, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=-1)

    assert x.shape[1] == max_len and y.shape[1] == max_len

    return x, y

def process_edf_arousal(edf_path, xml_path, save_dir, fs=50, pad=True, do_filter=False):
    raw = load_edf_file(edf_path, preload=True, resample=fs, do_filter=do_filter)
    meas_date = raw.info['meas_date']  # EDF start time
    data = raw.get_data()

    data_norm = moving_window_mean_rms_norm(data, fs=fs, window_min=18)
    x = data_norm.T  # (time, ch)
    x = x[:, :6] # EEG channels only

    events = load_arousal_xml(xml_path)
    total_samples = x.shape[0]
    y = create_arousal_labels_per_channel(events, meas_date, total_samples, sfreq=fs)
    for i in range(6):
        print(f"Channel {i}: {np.sum(y[i])} samples")
    
    if pad:
        x, y = pad_signals(x, y, 2**21)

    filename = basename(edf_path).replace(".edf", ".pkl")
    save_path = join(save_dir, filename)

    with open(save_path, "wb") as f:
        pickle.dump({"x": x, "y": y}, f)

    print(f"[Saved] {save_path}")

    return x, y


if __name__ == "__main__":
    sfreq = 100
    pad = True
    edf_dir = "/home/honeynaps/data/HN_DATA_AS/EDF"
    xml_dir = "/home/honeynaps/data/HN_DATA_AS/EBX/AROUS"
    save_dir = f"/home/honeynaps/data/HN_DATA_AS/PICKLE/ABS_SHIFT_{sfreq}"

    do_filter = True
    if pad:
        save_dir += "_PAD"

    if do_filter:
        save_dir += "_FILTERED"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]

    for i, edf_file in enumerate(edf_files):
        edf_path = os.path.join(edf_dir, edf_file)
        xml_path = os.path.join(xml_dir, edf_file.replace(".edf", ".xml"))

        try:
            process_edf_arousal(edf_path, xml_path, save_dir, pad=pad, fs=sfreq, do_filter=do_filter)
            print(f"Done processing {i+1}/{len(edf_files)}: {edf_file}")
        except Exception as e:
            print(f"Error: {edf_file}, {str(e)}")
            continue
