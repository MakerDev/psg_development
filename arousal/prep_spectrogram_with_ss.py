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

def create_sleep_stage_labels(events, total_samples, sfreq=50):
    SLEEPSTAGE_TO_LABEL = {
        "SLEEP-U":-1,
        "SLEEP-W":0, 
        "SLEEP-R":1,
        "SLEEP-1":2, 
        "SLEEP-2":3,
        "SLEEP-3":4,
        "SLEEP-WAKE":0,
        "SLEEP-REM":1,
        "SLEEP-N1":2,
        "SLEEP-N2":3,
        "SLEEP-N3":4,
    }

    y = np.zeros(total_samples, dtype=np.float32)

    for event in events:
        st, et = event["s_sec"], event["e_sec"]
        s_idx = int(st * sfreq)
        e_idx = int(et * sfreq)

        if s_idx < 0: s_idx = 0
        if e_idx > total_samples: e_idx = total_samples

        y[s_idx:e_idx] = SLEEPSTAGE_TO_LABEL[event["description"]]

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

def map_sleep_stage_label_to_spec_time(y_1d, t_array, fs=50, nperseg=100):
    half_win_sec = nperseg / (2.0 * fs)
    time_bins = len(t_array)

    label_spec_1d = np.zeros(time_bins, dtype=np.int32)
    
    for i, center_sec in enumerate(t_array):
        start_sec = center_sec - half_win_sec
        end_sec   = center_sec + half_win_sec
        
        start_idx = int(np.floor(start_sec * fs))
        end_idx   = int(np.ceil(end_sec * fs))

        # 범위를 실제 y_1d 범위 내로 클리핑
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(y_1d):
            end_idx = len(y_1d)

        # 해당 구간의 수면 단계 배열
        segment = y_1d[start_idx:end_idx]
        if len(segment) == 0:
            # 빈 구간이 생기면, 일단 0(또는 -1) 등으로 처리 가능
            label_spec_1d[i] = 0
            continue

        # 최빈값(majority vote)으로 대표 스테이지 결정
        unique_vals, counts = np.unique(segment, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_stage = unique_vals[majority_idx]

        label_spec_1d[i] = majority_stage

    return label_spec_1d


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
    label_2d = np.tile(label_1d, (freq_bins, 1))  # (F, T)
    return label_2d


def events_to_sec_events(events, meas_date):
    meas_date = meas_date.replace(tzinfo=None)
    sec_events = []
    for event in events:
        event["s_sec"] = (event["onset"] - meas_date).total_seconds()
        event["e_sec"] = event["s_sec"] + event["duration"]
        sec_events.append(event)
    return sec_events

def fill_unknown(events):
    n_unknown = 0

    for i in range(1, len(events)):
        curr_e = events[i]
        prev_e = events[i-1]

        if prev_e['e_sec'] != curr_e['s_sec']:
            raise Exception("There is a gap between events")
        
        if "U" in curr_e['description']:
            curr_e['description'] = prev_e['description']
            n_unknown += 1

    return events, n_unknown


def integrate_stage_info(x, y_sleep, num_stage=5):
    C, F, T = x.shape
    assert y_sleep.shape[0] == T, "y_cls와 x의 time축 길이가 달라서는 안 됩니다."

    stage_onehot = np.zeros((T, num_stage), dtype=np.float32)
    stage_onehot[np.arange(T, dtype=int), y_sleep] = 1.0

    stage_onehot = np.expand_dims(stage_onehot, axis=1)      # => (T, 1, num_stage)
    stage_onehot = np.tile(stage_onehot, (1, F, 1))          # => (T, freq, num_stage)

    stage_onehot = np.transpose(stage_onehot, (2,1,0))

    x_extended = np.concatenate([x, stage_onehot], axis=0)
    return x_extended


def process_edf_arousal_spec(edf_path, xml_path, save_dir, 
                             pad=True,
                             fs=50, 
                             nperseg=100, 
                             noverlap=50,
                             do_filter=False):
    sleep_xml = xml_path.replace("AROUS", "SLEEP")
    filename = basename(edf_path).replace(".edf", ".pkl")
    save_path = join(save_dir, filename)

    if os.path.exists(save_path):
        print(f"[Skip] {save_path}")
        return
    
    # ------ 1) EDF 로딩 & 정규화 ------
    raw = load_edf_file(edf_path, preload=True, resample=fs, do_filter=do_filter)
    meas_date = raw.info['meas_date'].replace(tzinfo=None)  # EDF start time
    data = raw.get_data()             # (channel, time)

    sleep_events = events_to_sec_events(load_sleep_stage(sleep_xml), meas_date)
    sleep_events, n_unknown = fill_unknown(sleep_events)
    first_sleep_onset = sleep_events[0]["onset"].replace(tzinfo=None)
    last_sleep_finish = sleep_events[-1]["onset"].replace(tzinfo=None) + timedelta(seconds=sleep_events[-1]["duration"])

    data_norm = moving_window_mean_rms_norm(data, fs=fs, window_min=18)
    x = data_norm  # shape: (channel, time)
    # x = data  # (time, channel)

    x = x.T  # => (time, channel)

    events = load_arousal_xml(xml_path)
    total_samples = x.shape[0]
    y = create_arousal_labels_extended(events, meas_date, total_samples, sfreq=fs)
    # y shape: (T,)

    # Crop x and y to start from the first sleep onset
    first_sleep_onset_sec = (first_sleep_onset - meas_date).total_seconds()
    first_sleep_onset_idx = int(first_sleep_onset_sec * fs)
    last_sleep_finish_sec = (last_sleep_finish - meas_date).total_seconds()
    last_sleep_finish_idx = int(last_sleep_finish_sec * fs)
    x = x[first_sleep_onset_idx:last_sleep_finish_idx, :]
    y = y[first_sleep_onset_idx:last_sleep_finish_idx]

    y_sleep = create_sleep_stage_labels(sleep_events, len(y), sfreq=fs)

    # ------ 4) 스펙트로그램 생성 ------
    spec, freqs, times = make_spectrogram(x, fs=fs, 
                                          nperseg=nperseg,
                                          noverlap=noverlap)
    # spec: (channel, freq_bins, time_bins)

    # ------ 5) 스펙트로그램 레이블 매핑 ------
    label_1d = map_label_to_spec_time(y, times, fs=fs, nperseg=nperseg)
    label_sleep_1d = map_sleep_stage_label_to_spec_time(y_sleep, times, fs=fs, nperseg=nperseg)
    # shape: (time_bins,)

    # 필요하다면 2D label (freq, time) => freq 축 동일
    label_2d = expand_label_freq(label_1d, freq_bins=len(freqs))
    # shape: (freq_bins, time_bins)

    spec = integrate_stage_info(spec, label_sleep_1d)


    # 원하는 형태로 dict 구성
    # - spec: (C, F, T)
    # - label: (F, T)  또는 (1, F, T) 로 만들어도 됨
    result = {
        "x": spec.astype(np.float32),
        "y": label_1d.astype(np.float32),
        "y_sleep": label_sleep_1d.astype(np.float32),
        "y_time": y,
        "freqs": freqs,     # (F,)
        "times": times,     # (T_spect,)
    }

    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[Saved Spec+Label] {save_path}")
    return spec, label_2d


def add_components(edf_path, xml_path, save_dir, fs=50):
    save_path = join(save_dir, basename(edf_path).replace(".edf", ".pkl"))

    with open(save_path, "rb") as f:
        result = pickle.load(f)

        times = result["times"]

    raw = load_edf_file(edf_path, preload=True, resample=fs)
    meas_date = raw.info['meas_date'].replace(tzinfo=None)  # EDF start time

    sleep_xml = xml_path.replace("AROUS", "SLEEP")
    sleep_xml = sleep_xml.replace("ASHIFT", "SLEEP")

    sleep_events = events_to_sec_events(load_sleep_stage(sleep_xml), meas_date)
    sleep_events, n_unknown = fill_unknown(sleep_events)
 
    y_sleep = create_sleep_stage_labels(sleep_events, len(result['y_time']), sfreq=fs)
    label_sleep_1d = map_sleep_stage_label_to_spec_time(y_sleep, times, fs=fs, nperseg=nperseg)

    result["meas_date"] = meas_date
    result["y_sleep"] = label_sleep_1d.astype(np.float32)
    result["y_sleep_time"] = y_sleep.astype(np.float32)

    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[Updated Components] {save_path}")


if __name__ == "__main__":
    sfreq = 50
    pad = True

    base_dir = "/home/honeynaps/data/GOLDEN"
    edf_dir = f"{base_dir}/EDF2"
    xml_dir = f"{base_dir}/EBX2/AROUS"

    # base_dir = "/home/honeynaps/data/dataset2"
    # edf_dir = f"{base_dir}/EDF"
    # xml_dir = f"{base_dir}/EBX/AROUS"

    base_dir = "/home/honeynaps/data/HN_DATA_AS"
    edf_dir = f"{base_dir}/EDF"
    xml_dir = f"{base_dir}/EBX/ASHIFT"

    tag = "tight"
    do_filter = False
    perseg_time = 1
    
    # 스펙트로그램용 저장 경로
    save_dir = f"{base_dir}/AROUS_SPEC/AROUSAL_SPEC_{sfreq}"
    if pad:
        save_dir += "_PAD"

    if do_filter:
        save_dir += "_FILTERED"

    # tag = "ss"
    # do_filter = False
    # # 스펙트로그램용 저장 경로
    # save_dir = f"{base_dir}/AROUS_SPEC/AROUSAL_SPEC_{sfreq}_PS{perseg_time}"
    # if pad:
    #     save_dir += "_PAD"

    # if do_filter:
    #     save_dir += "_FILTERED"

    if len(tag) > 0:
        save_dir += f"_{tag}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]

    nperseg = perseg_time * sfreq
    overlap = 0.5

    for i, edf_file in enumerate(edf_files):
        edf_path = os.path.join(edf_dir, edf_file)
        xml_path = os.path.join(xml_dir, edf_file.replace(".edf", "_AROUS.xml"))
        if "SCH_M_20_OV_230111R1_NO.pkl" in edf_file:
            print(f"Skip {edf_file}")
            continue

        try:
            # process_edf_arousal_spec(edf_path, xml_path, save_dir,
            #                          pad=pad,
            #                          fs=sfreq,
            #                          nperseg=nperseg,
            #                          noverlap=nperseg*overlap,
            #                          do_filter=do_filter)
            add_components(edf_path, xml_path, save_dir)
            print(f"Done processing {i+1}/{len(edf_files)}: {edf_file}")
        except Exception as e:
            print(f"Error: {edf_file}, {str(e)}")
            continue
