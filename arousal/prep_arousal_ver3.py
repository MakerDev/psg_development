import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import numpy as np
import datetime as dt
import xml.etree.ElementTree as ET
from datetime import timedelta
from os.path import basename, join 

import mne
from mne.io import read_raw_edf
from mne.filter import filter_data

from scipy.ndimage import uniform_filter1d
import random
from utils.tools import *

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

def create_arousal_labels_extended(events, meas_date, total_samples, sfreq=50):
    y = np.zeros(total_samples, dtype=np.float32)
    meas_date = meas_date.replace(tzinfo=None)

    for event in events:
        # 원래 arousal
        event_start = (event["onset"] - meas_date).total_seconds()
        event_end = event_start + event["duration"]
        
        # 확장: 시작 -2초, 끝 +10초
        extended_start = event_start - 2.0 + 2.0
        extended_end = event_end + 10.0    - 10.0

        # 음수 시작 방지
        if extended_start < 0:
            extended_start = 0

        s_idx = int(extended_start * sfreq)
        e_idx = int(extended_end * sfreq)

        # 범위 내로 자르기
        if s_idx < 0:
            s_idx = 0
        if e_idx > total_samples:
            e_idx = total_samples
        
        if s_idx < total_samples:
            y[s_idx:e_idx] = 1.0

    return y


def process_edf_arousal(edf_path, xml_path, save_dir, pad=True):
    raw = load_edf_file(edf_path, preload=True, resample=50)
    meas_date = raw.info['meas_date']  # EDF start time
    data = raw.get_data()

    data_norm = moving_window_mean_rms_norm(data, fs=50, window_min=18)
    x = data_norm.T  # (time, ch)

    events = load_arousal_xml(xml_path)
    total_samples = x.shape[0]
    y = create_arousal_labels_extended(events, meas_date, total_samples, sfreq=50)
    
    if pad:
        x, y = pad_signals(x, y, 2**21)

    filename = basename(edf_path).replace(".edf", ".pkl")
    save_path = join(save_dir, filename)

    with open(save_path, "wb") as f:
        pickle.dump({"x": x, "y": y}, f)

    print(f"[Saved] {save_path}")

    return x, y


if __name__ == "__main__":
    sfreq = 50
    pad = True
    edf_dir = "/home/honeynaps/data/GOLDEN/EDF2"
    xml_dir = "/home/honeynaps/data/GOLDEN/EBX2/AROUS"
    save_dir = f"/home/honeynaps/data/GOLDEN/PICKLE/AROUSAL_VER3_NOEXT_{sfreq}"

    if pad:
        save_dir += "_PAD"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]

    for i, edf_file in enumerate(edf_files):
        edf_path = os.path.join(edf_dir, edf_file)
        xml_path = os.path.join(xml_dir, edf_file.replace(".edf", "_AROUS.xml"))

        try:
            process_edf_arousal(edf_path, xml_path, save_dir, pad)
            print(f"Done processing {i+1}/{len(edf_files)}: {edf_file}")
        except Exception as e:
            print(f"Error: {edf_file}, {str(e)}")
            continue
