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
import multiprocessing
import concurrent.futures

# ==============================
# 0) 필요한 함수/유틸 불러오기 (가정)
# ==============================
def load_edf_file(edf_path, preload=True, resample=50):
    """
    EDF 파일 로드 및 리샘플링. (사용자 함수 예시)
    """
    raw = read_raw_edf(edf_path, preload=preload, verbose=False)
    if resample is not None:
        raw.resample(resample)
    return raw

def load_arousal_xml(xml_path):
    """
    XML에서 arousal 이벤트 파싱. (사용자 함수 예시)
    이벤트 리스트를 [{'onset': datetime, 'duration': float}, ...] 형태로 반환
    """
    events = []
    if not os.path.exists(xml_path):
        return events  # 없으면 빈 리스트
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for event in root.findall(".//ScoredEvent"):
        if event.find("EventType").text == "Arousal":
            onset_str = event.find("Start").text  # '10.0' (초)
            duration_str = event.find("Duration").text  # '3.0' (초)
            # 실제 EDF start time이 meas_date라고 가정하고, offset 계산이 필요하다면 적절히 수행
            onset_time = dt.datetime(2000,1,1) + dt.timedelta(seconds=float(onset_str)) 
            events.append({
                "onset": onset_time,
                "duration": float(duration_str)
            })
    return events

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
        # 0에 대한 안정 처리
        rms_val[rms_val < 1e-12] = 1e-12
        out[ch_idx] = (ch_data - mean_val) / rms_val

    return out

# ==============================
# 2) arousal 레이블 생성 (기존 동일)
# ==============================
def create_arousal_labels_extended(events, meas_date, total_samples, sfreq=50):
    y = np.zeros(total_samples, dtype=np.float32)
    # meas_date를 naive로 처리
    meas_date = meas_date.replace(tzinfo=None) if meas_date is not None else dt.datetime(2000,1,1)

    for event in events:
        event_start = (event["onset"] - meas_date).total_seconds()
        event_end = event_start + event["duration"]
        
        # 확장 범위 예시 (원래 코드와 동일)
        extended_start = event_start - 2.0 + 2.0
        extended_end = event_end + 10.0 - 10.0

        if extended_start < 0:
            extended_start = 0

        s_idx = int(extended_start * sfreq)
        e_idx = int(extended_end * sfreq)

        if s_idx < 0: s_idx = 0
        if e_idx > total_samples: e_idx = total_samples

        if s_idx < total_samples:
            y[s_idx:e_idx] = 1.0

    return y

# ==============================
# 3) Spectrogram + 레이블 매핑 부분 (기존 동일)
# ==============================
def make_spectrogram(data, fs=50, nperseg=100, noverlap=50):
    """
    data: (time, channel)
    fs: 샘플링 레이트(Hz)
    nperseg, noverlap: STFT 파라미터
    return:
      spec: (channel, freq_bins, time_bins)
      freqs: (freq_bins,)
      times: (time_bins,) => spectrogram 윈도우의 중심(초)
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
        # log 변환
        Sxx_log = np.log1p(Sxx)
        specs.append(Sxx_log[None, ...])  # (1, freq_bins, time_bins)

    spec = np.concatenate(specs, axis=0)  # (C, freq_bins, time_bins)
    return spec, f, t

def map_label_to_spec_time(y_1d, t_array, fs=50, nperseg=100):
    """
    y_1d : (T,) 원본 시계열 레이블 (0/1)
    t_array : spectrogram의 time_bins (윈도우 중심시간, sec)
    fs : 샘플링 레이트
    nperseg : STFT 윈도우 크기(샘플)
    
    return:
      label_spec_1d : (time_bins,) 
        각 time bin마다 0/1 라벨
    """
    half_win_sec = nperseg / (2.0 * fs)  # 윈도우 절반 길이(초 단위)
    time_bins = len(t_array)

    label_spec_1d = np.zeros(time_bins, dtype=np.float32)
    
    for i, center_sec in enumerate(t_array):
        start_sec = center_sec - half_win_sec
        end_sec   = center_sec + half_win_sec
        
        start_idx = int(np.floor(start_sec * fs))
        end_idx   = int(np.ceil(end_sec * fs))

        if start_idx < 0:
            start_idx = 0
        if end_idx > len(y_1d):
            end_idx = len(y_1d)

        # 구간 내에 1이 하나라도 있으면 1
        if np.any(y_1d[start_idx:end_idx] == 1):
            label_spec_1d[i] = 1.0

    return label_spec_1d

def expand_label_freq(label_1d, freq_bins):
    """
    label_1d: (time_bins,)
    freq_bins: int
    
    return: (freq_bins, time_bins)
       freq 방향으로 똑같이 복제
    """
    label_2d = np.tile(label_1d, (freq_bins, 1))  # (F, T)
    return label_2d

# ==============================
# 4) 실제 EDF -> Spectrogram 변환 함수
# ==============================
def process_edf_arousal_spec(edf_path, xml_path, save_dir, 
                             pad=True,
                             fs=50, 
                             nperseg=100, 
                             noverlap=50):
    """
    - EDF를 로딩하여, 
      1) moving_window_mean_rms_norm 정규화
      2) arousal 레이블 생성 (y)
      3) 스펙트로그램 생성 (channel, freq, time)
      4) time bin에 맞춘 레이블(1D/2D)
      5) pickle 저장
    """
    # 1) EDF 로딩 및 정규화
    raw = load_edf_file(edf_path, preload=True, resample=fs)
    meas_date = raw.info['meas_date']  # EDF start time
    data = raw.get_data()  # (channel, time)

    data_norm = moving_window_mean_rms_norm(data, fs=fs, window_min=18)
    x = data_norm  # (channel, time)
    x = x.T        # => (time, channel)
    
    # 2) arousal 레이블 생성
    events = load_arousal_xml(xml_path)
    total_samples = x.shape[0]
    y = create_arousal_labels_extended(events, meas_date, total_samples, sfreq=fs)  # (T,)

    # (옵션) pad 처리가 필요하다면 여기서 추가 (현재 pad 파라미터는 예시로만 존재)

    # 3) 스펙트로그램 생성
    spec, freqs, times = make_spectrogram(x, fs=fs, 
                                          nperseg=nperseg,
                                          noverlap=noverlap)

    # 4) 스펙트로그램 레이블
    label_1d = map_label_to_spec_time(y, times, fs=fs, nperseg=nperseg)
    label_2d = expand_label_freq(label_1d, freq_bins=len(freqs))

    # 5) 결과 저장 (pickle)
    filename = basename(edf_path).replace(".edf", ".pkl")
    save_path = join(save_dir, filename)

    result = {
        "x": spec.astype(np.float32),  # (C, F, T)
        "y": label_1d.astype(np.float32),
        "y_time": y,
        "freqs": freqs,   
        "times": times,
    }

    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[Saved Spec+Label] {save_path}")
    return save_path  # 반환값 예시


# ==============================
# 5) 각 파일 처리 함수(래퍼)
# ==============================
def process_single_file(edf_file, edf_dir, xml_dir, save_dir,
                        pad=True,
                        fs=50,
                        nperseg=50,
                        noverlap=25):
    """
    병렬로 실행할 개별 작업 함수.
    """
    edf_path = os.path.join(edf_dir, edf_file)
    xml_path = os.path.join(xml_dir, edf_file.replace(".edf", "_AROUS.xml"))
    try:
        save_path = process_edf_arousal_spec(edf_path, xml_path, save_dir,
                                             pad=pad,
                                             fs=fs,
                                             nperseg=nperseg,
                                             noverlap=noverlap)
        return (edf_file, None)  # None은 에러 없음을 의미
    except Exception as e:
        return (edf_file, str(e))  # 에러 내용을 리턴


# ==============================
# 6) 메인: 병렬 실행
# ==============================
if __name__ == "__main__":
    sfreq = 50
    pad = True

    base_dir = "/home/honeynaps/data/GOLDEN"
    # base_dir = "/home/honeynaps/data/dataset2"

    edf_dir = f"{base_dir}/EDF2"
    xml_dir = f"{base_dir}/EBX2/AROUS"
    tag = "loose"
    
    # 스펙트로그램용 저장 경로
    save_dir = f"{base_dir}/AROUS_SPEC/AROUSAL_SPEC_{sfreq}"
    if pad:
        save_dir += "_PAD"
    save_dir += f"_{tag}" if tag else ""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]

    # 병렬 처리할 워커(프로세스) 수
    # - CPU 코어 수만큼 사용 권장. (필요시 조정)
    num_workers = multiprocessing.cpu_count()

    # ProcessPoolExecutor를 이용하여 병렬 처리
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 각 edf_file을 parallel하게 처리하도록 future 생성
        futures = {
            executor.submit(
                process_single_file,
                edf_file, edf_dir, xml_dir, save_dir, 
                pad, sfreq, 150, 75
            ): edf_file
            for edf_file in edf_files
        }

        # 처리 완료된 순서대로 결과 출력
        for future in concurrent.futures.as_completed(futures):
            edf_file = futures[future]
            try:
                file_name, error_msg = future.result()
                if error_msg is None:
                    print(f"[OK] {file_name} 처리 완료.")
                else:
                    print(f"[ERROR] {file_name} 처리 중 오류: {error_msg}")
            except Exception as e:
                print(f"[EXCEPT] {edf_file} 처리 중 예외 발생: {e}")
