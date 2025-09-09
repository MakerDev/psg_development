import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import argparse
from os import path
from datetime import timedelta
import pandas as pd
import pickle
import os
import datetime as dt
import xml.etree.ElementTree as ET
import pyedflib
from xml.dom import minidom
from datetime import datetime, timedelta

import mne 
from mne.io import read_raw_edf
from os.path import basename, join 

from scipy.signal import find_peaks, hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from mne.filter import filter_data
from utils.tools import *

def band_pass_filter(x, lf, hf, fs=100, tb='auto'):

    return filter_data(
        x, sfreq=fs, l_freq=lf, h_freq=hf, verbose=False,
        l_trans_bandwidth=tb, h_trans_bandwidth=tb)

def moving_window_mean_rms_norm_fast(x, fs=100, window_min=18):
    window_size = int(window_min * 60 * fs)  # 18분 * 60초 * fs
    
    out = np.zeros_like(x, dtype=np.float32)

    for ch_idx in range(x.shape[0]):
        ch_data = x[ch_idx]
        # 1) 이동 평균
        ch_mean = uniform_filter1d(ch_data, size=window_size, mode='reflect')
        # 2) 이동 RMS
        ch_sqr = ch_data**2
        ch_rms = np.sqrt(
            uniform_filter1d(ch_sqr, size=window_size, mode='reflect')
        )
        ch_rms[ch_rms < 1e-12] = 1e-12
        out[ch_idx] = (ch_data - ch_mean) / ch_rms

    return out

def chunkwise_mean_rms_norm(x, fs=100, window_min=18):
    window_size = int(window_min * 60 * fs)  # 18분 * 60초 * fs
    out = np.zeros_like(x, dtype=np.float32)

    for ch_idx in range(x.shape[0]):
        ch_data = x[ch_idx]
        length = ch_data.shape[0]

        start_idx = 0
        while start_idx < length:
            end_idx = min(start_idx + window_size, length)
            chunk = ch_data[start_idx:end_idx]

            mean_val = np.mean(chunk)
            rms_val = np.sqrt(np.mean(chunk**2))
            if rms_val < 1e-12:
                rms_val = 1e-12

            out[ch_idx, start_idx:end_idx] = (chunk - mean_val) / rms_val

            start_idx = end_idx  # 오버랩 없이 다음 구간으로 이동

    return out

def prep_psg_signal(x, transpose=True, fs=50):
    # EEG (0~5)
    eeg = np.array([
        band_pass_filter(x=x[i], lf=0.5, hf=24.999, fs=fs)
        for i in range(6)
    ])  
    # EOG (6,7)
    eog = np.array([
        band_pass_filter(x=x[i], lf=0.5, hf=24.999, fs=fs)
        for i in range(6,8)
    ])  
    # EMG (8)
    emg = band_pass_filter(x=x[-1], lf=0.5, hf=24.999, fs=fs)[np.newaxis,:]

    x_filtered = np.vstack([eeg, eog, emg])

    x_norm = chunkwise_mean_rms_norm(x_filtered, fs=fs, window_min=18)

    if transpose:
        x_norm = x_norm.T

    return x_norm


def get_events_from_labels(labels):
    events = []
    in_event = False
    start = 0
    for i, val in enumerate(labels):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            in_event = False
            end = i-1
            events.append((start, end))
    if in_event:
        events.append((start, len(labels)-1))
    return events


def process_edf_arousal_to_pickle(edf_path, xml_path, save_dir="./output", sfreq=100, xml_sample_dir=None, pad=True):
    raw = load_edf_file(path=edf_path, preload=True, resample=sfreq, preset="STAGENET", exclude=True, missing_ch='handling')
    
    events = load_arousal_xml(xml_path)
    
    meas_date = raw.info['meas_date']
    data = raw.get_data()

    x = prep_psg_signal(data, transpose=True, fs=sfreq)  # (time, ch)
    
    total_samples = x.shape[0]
    y = create_arousal_labels(events, meas_date, total_samples, sfreq=sfreq)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = basename(edf_path).replace(".edf", ".pkl")
    save_path = join(save_dir, filename)

    if xml_sample_dir:
        xml_sample_path = join(xml_sample_dir, filename.replace(".pkl", "_AROUS.xml"))
        save_arousal_xml(meas_date, y, sfreq, xml_sample_path)

    if pad:
        max_len = 2**22 if sfreq == 100 else 2**21
        x, y = pad_signals(x, y, max_len)
        save_path = save_path.replace("AROUSAL_NEW", "AROUSAL_NEW_PAD")

    with open(save_path, "wb") as f:
        pickle.dump({"x": x, "y": y}, f)

    print("Saved:", save_path)




if __name__ == "__main__":
    sfreq = 50
    pad = True
    edf_dir = "/home/honeynaps/data/HN_DATA_AS/EDF"
    xml_dir = "/home/honeynaps/data/HN_DATA_AS/EBX/AROUS" 
    save_dir = f"/home/honeynaps/data/HN_DATA_AS/PICKLE/AROUSAL_VER2_{sfreq}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pad:
        save_dir += "_PAD"

    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]

    for i, edf_file in enumerate(edf_files):
        edf_path = join(edf_dir, edf_file)
        xml_path = join(xml_dir, edf_file.replace(".edf", "_AROUS.xml"))

        try:
            process_edf_arousal_to_pickle(edf_path, xml_path, save_dir, sfreq, None, pad)
            print(f"Done processing {i+1}/{len(edf_files)}: {edf_file}")
        except Exception as e:
            print(f"Error: {edf_file}, {str(e)}")
            continue
