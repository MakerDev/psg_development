import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import numpy as np
from os.path import basename, join 
from scipy.ndimage import uniform_filter1d
from scipy.signal import spectrogram
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


def process_edf_arousal_spec(edf_path, xml_path, save_dir, 
                             do_filter=True,
                             fs=50, 
                             nperseg=100, 
                             noverlap=50):
    raw = load_edf_file(edf_path, preload=True, resample=fs, do_filter=do_filter)
    meas_date = raw.info['meas_date']  # EDF start time
    data = raw.get_data()

    data_norm = moving_window_mean_rms_norm(data, fs=fs, window_min=18)
    x = data_norm.T  # (time, ch)
    x = x[:, :6] # EEG channels only

    events = load_arousal_xml(xml_path)
    total_samples = x.shape[0]
    y = create_arousal_labels_per_channel(events, meas_date, total_samples, sfreq=fs)

    spec, freqs, times = make_spectrogram(x, fs=fs, 
                                          nperseg=nperseg,
                                          noverlap=noverlap)

    for i in range(6):
        print(f"Channel {i}: {np.sum(y[i])} samples")

    y_mapped = np.zeros((6, len(times)), dtype=np.float32)
    for i in range(6):
        y_mapped[i] = map_label_to_spec_time(y[i], times, fs=fs, nperseg=nperseg)


    filename = basename(edf_path).replace(".edf", ".pkl")
    save_path = join(save_dir, filename)
    result = {
        "x": spec.astype(np.float32),
        "y": y_mapped.astype(np.float32),
        "y_time": y,
        "freqs": freqs,
        "times": times,
    }

    with open(save_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[Saved] {save_path}")

    return x, y_mapped


if __name__ == "__main__":
    sfreq = 50
    do_filter = False
    edf_dir = "/home/honeynaps/data/HN_DATA_AS/EDF"
    xml_dir = "/home/honeynaps/data/HN_DATA_AS/EBX/ASHIFT"
    save_dir = f"/home/honeynaps/data/HN_DATA_AS/AROUS_SPEC/ABS_SHIFT_{sfreq}_new"
    
    edf_dir = "/home/honeynaps/data/HN_DATA_AS/250428/EDF"
    xml_dir = "/home/honeynaps/data/HN_DATA_AS/250428/EBX/ASHIFT"
    save_dir = f"/home/honeynaps/data/HN_DATA_AS/250428/AROUS_SPEC/ABS_SHIFT_{sfreq}_new"

    if do_filter:
        save_dir += "_filtered"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]
    perseg = 1
    nperseg = perseg * sfreq
    overlap = 0.5

    for i, edf_file in enumerate(edf_files):
        edf_path = os.path.join(edf_dir, edf_file)
        xml_path = os.path.join(xml_dir, edf_file.replace(".edf", ".xml"))

        try:
            process_edf_arousal_spec(edf_path, xml_path, save_dir,
                                     fs=sfreq,
                                     do_filter=do_filter,
                                     nperseg=nperseg,
                                     noverlap=nperseg * overlap)            
            print(f"Done processing {i+1}/{len(edf_files)}: {edf_file}")
        except Exception as e:
            print(f"Error: {edf_file}, {str(e)}")
            continue
