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
from scipy.signal import spectrogram

import random
from utils.tools import *

# ==============================
# 1) moving_window_mean_rms_norm (기존 동일)
# ==============================
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

# ==============================
# 2) arousal 레이블 생성 (기존 동일)
# ==============================
def create_arousal_labels_extended(events, meas_date, total_samples, sfreq=50):
    y = np.zeros(total_samples, dtype=np.float32)
    meas_date = meas_date.replace(tzinfo=None)

    for event in events:
        event_start = (event["onset"] - meas_date).total_seconds()
        event_end = event_start + event["duration"]
        
        # 확장 범위 예시 (원래 코드와 동일)
        extended_start = event_start 
        extended_end = event_end

        if extended_start < 0:
            extended_start = 0

        s_idx = int(extended_start * sfreq)
        e_idx = int(extended_end * sfreq)

        if s_idx < 0: s_idx = 0
        if e_idx > total_samples: e_idx = total_samples

        if s_idx < total_samples:
            y[s_idx:e_idx] = 1.0

    return y


def make_spectrogram(data, fs=50, nperseg=100, noverlap=50):
    # data는 (T, C) 형태
    T, C = data.shape

    if C > 10:
        C, T = data.shape
    
    specs = []
    for ch in range(C):
        f, t, Sxx = spectrogram(
            data[:, ch],
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nperseg,
            scaling='density',
            mode='psd'
        )
        Sxx_log = np.log1p(Sxx)
        specs.append(Sxx_log[None, ...])  # (1, freq_bins, time_bins)

    spec = np.concatenate(specs, axis=0)  # (C, freq_bins, time_bins)
    return spec, f, t


def map_label_to_spec_time(y_1d, t_array, fs=50, nperseg=100):
    half_win_sec = nperseg / (2.0 * fs)  # 윈도우 절반 길이(초 단위)
    time_bins = len(t_array)

    label_spec_1d = np.zeros(time_bins, dtype=np.float32)
    
    for i, center_sec in enumerate(t_array):
        start_sec = center_sec - half_win_sec
        end_sec   = center_sec + half_win_sec
        
        # 실제 인덱스
        start_idx = int(np.floor(start_sec * fs))
        end_idx   = int(np.ceil(end_sec * fs))

        # 범위 클리핑
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(y_1d):
            end_idx = len(y_1d)

        # 구간 내에 1이 하나라도 있으면 1
        if np.any(y_1d[start_idx:end_idx] == 1):
            label_spec_1d[i] = 1.0
        else:
            label_spec_1d[i] = 0.0

    return label_spec_1d


def expand_label_freq(label_1d, freq_bins):
    """
    label_1d: (time_bins,)
    freq_bins: int
    
    return: (freq_bins, time_bins)
       freq 방향으로 똑같이 복제 (주파수 축 전체 같은 라벨)
    """
    label_2d = np.tile(label_1d, (freq_bins, 1))  # (F, T)
    return label_2d


def process_edf_arousal_spec(edf_path, xml_path, save_dir, 
                             pad=True,
                             fs=50, 
                             nperseg=100, 
                             noverlap=50,
                             do_filter=False):
    filename = basename(edf_path).replace(".edf", ".pkl")
    save_path = join(save_dir, filename)

    # if os.path.exists(save_path):
    #     print(f"[Skip] {save_path}")
    #     return

    # ------ 1) EDF 로딩 & 정규화 ------
    raw = load_edf_file(edf_path, preload=True, resample=fs, do_filter=do_filter)
    meas_date = raw.info['meas_date']  # EDF start time
    data = raw.get_data()             # (channel, time)

    data_norm = moving_window_mean_rms_norm(data, fs=fs, window_min=18)
    x = data_norm  # shape: (channel, time)
    # x = data  # (time, channel)

    x = x.T  # => (time, channel)

    events = load_arousal_xml(xml_path)
    total_samples = x.shape[0]
    y = create_arousal_labels_extended(events, meas_date, total_samples, sfreq=fs)
    # y shape: (T,)

    # ------ 4) 스펙트로그램 생성 ------
    spec, freqs, times = make_spectrogram(x, fs=fs, 
                                          nperseg=nperseg,
                                          noverlap=noverlap)
    # spec: (channel, freq_bins, time_bins)

    # ------ 5) 스펙트로그램 레이블 매핑 ------
    label_1d = map_label_to_spec_time(y, times, fs=fs, nperseg=nperseg)
    # shape: (time_bins,)

    # 필요하다면 2D label (freq, time) => freq 축 동일
    label_2d = expand_label_freq(label_1d, freq_bins=len(freqs))
    # shape: (freq_bins, time_bins)

    # ------ 6) 저장 ------
    threshold = 1e-5
    per_ch_mask = np.all(spec < threshold, axis=1)
    artifact_mask = np.any(per_ch_mask, axis=0)
    # valid_idx    = ~artifact_mask
    label_1d[artifact_mask] = -1


    # Fill here

    # 원하는 형태로 dict 구성
    # - spec: (C, F, T)
    # - label: (F, T)  또는 (1, F, T) 로 만들어도 됨
    result = {
        "x": spec.astype(np.float32),
        "y": label_1d.astype(np.float32),
        "y_time": y,
        "freqs": freqs,     # (F,)
        "times": times,     # (T_spect,)
        "meas_date": meas_date,  # (datetime)
    }

    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[Saved Spec+Label] {save_path}")
    return spec, label_2d


def add_components(edf_path, xml_path, save_dir, fs=50):
    raw = load_edf_file(edf_path, preload=True, resample=fs)
    meas_date = raw.info['meas_date']  # EDF start time

    save_path = join(save_dir, basename(edf_path).replace(".edf", ".pkl"))

    with open(save_path, "rb") as f:
        result = pickle.load(f)

    result["meas_date"] = meas_date

    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[Updated Components] {save_path}")

# ==============================
# 5) 실행부
# ==============================
if __name__ == "__main__":
    sfreq = 50
    pad = True

    base_dir = "/home/honeynaps/data/GOLDEN"
    edf_dir = f"{base_dir}/EDF2"
    xml_dir = f"{base_dir}/EBX2/AROUS"

    # base_dir = "/home/honeynaps/data/dataset2"
    # edf_dir = f"{base_dir}/EDF"
    # xml_dir = f"{base_dir}/EBX/AROUS"

    # base_dir = "/home/honeynaps/data/HN_DATA_AS"
    # edf_dir = f"{base_dir}/EDF"
    # xml_dir = f"{base_dir}/EBX/ASHIFT"

    # base_dir = "/home/honeynaps/data/GOLDEN2"
    # edf_dir = f"{base_dir}/EDF"
    # xml_dir = f"{base_dir}/TRUE/AROUS"

    tag = "artifact"
    do_filter = False
    perseg = 1
    
    # 스펙트로그램용 저장 경로
    save_dir = f"{base_dir}/AROUS_SPEC/AROUSAL_SPEC_{sfreq}"
    if pad:
        save_dir += "_PAD"

    if do_filter:
        save_dir += "_FILTERED"

    # save_dir += f"_PS{perseg}"

    if len(tag) > 0:
        save_dir += f"_{tag}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]

    nperseg = perseg * sfreq
    overlap = 0.5

    swap = "AROUS" if "HN_DATA" not in base_dir else "ASHIFT"

    for i, edf_file in enumerate(edf_files):
        edf_path = os.path.join(edf_dir, edf_file)
        xml_path = os.path.join(xml_dir, edf_file.replace(".edf", f"_{swap}.xml"))

        try:
            process_edf_arousal_spec(edf_path, xml_path, save_dir,
                                     pad=pad,
                                     fs=sfreq,
                                     nperseg=nperseg,
                                     noverlap=nperseg*overlap,
                                     do_filter=do_filter)
            # add_components(edf_path, xml_path, save_dir)
            print(f"Done processing {i+1}/{len(edf_files)}: {edf_file}")
        except Exception as e:
            print(f"Error: {edf_file}, {str(e)}")
            continue
