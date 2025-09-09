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
from utils.tools import *  # 사용 중인 함수들 (ex: load_arousal_xml 등)

# ==============================================
# 1) moving_window_mean_rms_norm (수정: window_min=2)
# ==============================================
def moving_window_mean_rms_norm(x, fs=50, window_min=2):
    """
    x: shape (channel, time)
    fs: 샘플링 레이트
    window_min: 이동 윈도우 길이(분)

    더 짧은 윈도우(예: 2분)로 정규화 -> 장기 드리프트는 제거하되, 
    너무 긴 스케일 상실을 방지.
    """
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

# ==============================================
# 2) arousal 레이블 생성 (기존 동일)
# ==============================================
def create_arousal_labels_extended(events, meas_date, total_samples, sfreq=50):
    y = np.zeros(total_samples, dtype=np.float32)
    meas_date = meas_date.replace(tzinfo=None)

    for event in events:
        event_start = (event["onset"] - meas_date).total_seconds()
        event_end = event_start + event["duration"]
        
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

# ==============================================
# 3) STFT (스펙트로그램) (수정: nperseg=2s, noverlap=1s)
# ==============================================
def make_spectrogram(data, fs=50, nperseg=100, noverlap=50):
    """
    data: shape (time, C)
          만약 shape (C, time)이면 코드 내에서 전치 필요.

    nperseg=2*fs => 2초 창
    noverlap=1*fs => 1초 중첩
    """
    T, C = data.shape
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
        # 로그 변환
        Sxx_log = np.log1p(Sxx)
        specs.append(Sxx_log[None, ...])  # (1, freq_bins, time_bins)

    spec = np.concatenate(specs, axis=0)  # (C, freq_bins, time_bins)
    return spec, f, t

def map_label_to_spec_time(y_1d, t_array, fs=50, nperseg=100):
    half_win_sec = nperseg / (2.0 * fs)
    time_bins = len(t_array)
    label_spec_1d = np.zeros(time_bins, dtype=np.float32)
    
    for i, center_sec in enumerate(t_array):
        start_sec = center_sec - half_win_sec
        end_sec   = center_sec + half_win_sec
        
        start_idx = int(np.floor(start_sec * fs))
        end_idx   = int(np.ceil(end_sec * fs))
        if start_idx < 0: start_idx = 0
        if end_idx > len(y_1d): end_idx = len(y_1d)

        # 구간 내에 1이 하나라도 있으면 1
        if np.any(y_1d[start_idx:end_idx] == 1):
            label_spec_1d[i] = 1.0
        else:
            label_spec_1d[i] = 0.0
    return label_spec_1d

def expand_label_freq(label_1d, freq_bins):
    label_2d = np.tile(label_1d, (freq_bins, 1))  # (F, T)
    return label_2d



# ==============================================
# 5) 최종 전처리 + 스펙트로그램 생성
# ==============================================
def process_edf_arousal_spec_best(edf_path, xml_path, save_dir,
                                  fs=50, 
                                  nperseg=100, 
                                  noverlap=50,
                                  ica_remove=False,
                                  window_min=2):
    filename = basename(edf_path).replace(".edf", ".pkl")
    save_path = join(save_dir, filename)

    if os.path.exists(save_path):
        print(f"[Skip existing] {save_path}")
        return

    # 1) EDF 로드
    raw = load_edf_file(edf_path, preload=True, resample=fs, ica_remove=ica_remove)
    meas_date = raw.info['meas_date']
    data = raw.get_data()  # shape: (n_channels, time)

    # 2) moving window norm
    data_norm = moving_window_mean_rms_norm(data, fs=fs, window_min=window_min)
    # (channel, time)

    # transpose => (time, channel)
    x = data_norm.T
    total_samples = x.shape[0]

    # 3) arousal 레이블
    events = load_arousal_xml(xml_path)  # 유저 기존 함수
    y = create_arousal_labels_extended(events, meas_date, total_samples, sfreq=fs)

    # 4) 스펙트로그램
    spec, freqs, times = make_spectrogram(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap
    )  
    # spec: (channel, freq_bins, time_bins)

    # 5) 레이블 -> 스펙트로그램 시각으로 맵핑
    label_1d = map_label_to_spec_time(y, times, fs=fs, nperseg=nperseg)

    # 필요시 freq x time label
    # label_2d = expand_label_freq(label_1d, freq_bins=len(freqs))

    # 6) dict 저장
    result = {
        "x": spec.astype(np.float32),   # shape (C, F, T)
        "y": label_1d.astype(np.float32),   # shape (T_spect,)
        "y_time": y.astype(np.float32),     # shape (time_samples,)
        "freqs": freqs,     
        "times": times,     
        "meas_date": meas_date,
    }

    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[Saved best-preproc Spec] {save_path}")


# ==============================================
# 6) 실행부 예시
# ==============================================
if __name__ == "__main__":
    fs = 50

    base_dir = "/home/honeynaps/data/GOLDEN"
    edf_dir = f"{base_dir}/EDF2"
    xml_dir = f"{base_dir}/EBX2/AROUS"

    base_dir = "/home/honeynaps/data/dataset2"
    edf_dir = f"{base_dir}/EDF"
    xml_dir = f"{base_dir}/EBX/AROUS"

    # base_dir = "/home/honeynaps/data/HN_DATA_AS"
    # edf_dir = f"{base_dir}/EDF"
    # xml_dir = f"{base_dir}/EBX/AROUS"

    perseg_time = 1
    save_dir = f"{base_dir}/AROUS_SPEC/AROUSAL_SPEC_{fs}_PS{perseg_time}"
    ica_remove = True
    tag = "opt"

    if ica_remove:
        tag+="_ica"


    if "HN_DATA" in base_dir:
        sub = ""
    else:
        sub = "_AROUS"

    if len(tag) > 0:
        save_dir += f"_{tag}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]

    # STFT 파라미터: 2초 윈도우, 1초 오버랩
    nperseg = perseg_time * fs
    noverlap = 0.5 * nperseg

    for i, edf_file in enumerate(edf_files):
        edf_path = os.path.join(edf_dir, edf_file)
        xml_path = os.path.join(xml_dir, edf_file.replace(".edf", f"{sub}.xml"))

        try:
            process_edf_arousal_spec_best(
                edf_path, xml_path, save_dir,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                ica_remove=ica_remove,
                window_min=2  # 정규화 2분
            )
            print(f"Done processing {i+1}/{len(edf_files)}: {edf_file}")
        except Exception as e:
            print(f"Error with {edf_file}: {str(e)}")
            continue
