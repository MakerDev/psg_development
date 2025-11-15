"""
Multimodal Arousal Preprocessing
- Combines time domain (raw signal + amplitude features) and frequency domain (spectrogram)
- Time domain captures abrupt amplitude changes
- Frequency domain captures frequency shifts
"""

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
from scipy.signal import spectrogram, hilbert, butter, sosfilt
from scipy.stats import skew, kurtosis

import random
from utils.tools import *


def robust_scale(x, fs=50):
    """Robust scaling using median and MAD"""
    median = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - median), axis=1, keepdims=True) + 1e-9
    return (x - median) / mad


def moving_window_mean_rms_norm(x, fs=50, window_min=18):
    """Moving window normalization"""
    window_size = int(window_min * 60 * fs)
    out = np.zeros_like(x, dtype=np.float32)

    for ch_idx in range(x.shape[0]):
        ch_data = x[ch_idx]
        mean_val = uniform_filter1d(ch_data, size=window_size, mode='reflect')
        sqr_val = ch_data**2
        rms_val = np.sqrt(uniform_filter1d(sqr_val, size=window_size, mode='reflect'))
        rms_val[rms_val < 1e-12] = 1e-12
        out[ch_idx] = (ch_data - mean_val) / rms_val

    return out


def extract_amplitude_features(data, fs=50):
    """
    Extract amplitude-based features for arousal detection
    Args:
        data: (channels, time) array
        fs: sampling frequency
    Returns:
        features: (channels, n_features, time) array
    """
    n_channels, n_samples = data.shape

    # 1. Signal envelope using Hilbert transform
    envelope = np.abs(hilbert(data, axis=1))

    # 2. First derivative (rate of change)
    gradient = np.gradient(data, axis=1) * fs

    # 3. Absolute amplitude
    abs_amplitude = np.abs(data)

    # 4. Smoothed envelope (captures slower amplitude changes)
    window_size = int(0.5 * fs)  # 0.5 second window
    smoothed_envelope = uniform_filter1d(envelope, size=window_size, mode='reflect', axis=1)

    # 5. High-frequency energy (alpha-beta band activity)
    # Use butterworth bandpass filter for alpha-beta band (12-24 Hz)
    # Note: For fs=50Hz, Nyquist freq is 25Hz, so we use 24Hz as upper bound
    # This captures arousal-related frequency increases
    sos = butter(4, [12, 24], btype='bandpass', fs=fs, output='sos')
    hf_filtered = sosfilt(sos, data, axis=1)
    hf_energy = hf_filtered ** 2
    hf_energy_smooth = uniform_filter1d(hf_energy, size=window_size, mode='reflect', axis=1)

    # Stack features: [raw, envelope, gradient, abs_amp, smooth_env, hf_energy]
    # Shape: (channels, 6, time)
    features = np.stack([
        data,
        envelope,
        gradient,
        abs_amplitude,
        smoothed_envelope,
        hf_energy_smooth
    ], axis=1)

    return features.astype(np.float32)


def extract_statistical_features(data, fs=50, window_sec=1.0):
    """
    Extract statistical features over sliding windows
    Args:
        data: (channels, time) array
        fs: sampling frequency
        window_sec: window size in seconds
    Returns:
        features: (channels, n_features, n_windows) array
    """
    n_channels, n_samples = data.shape
    window_size = int(window_sec * fs)
    hop_size = window_size // 2  # 50% overlap

    n_windows = (n_samples - window_size) // hop_size + 1

    # Features: mean, std, min, max, skewness, kurtosis
    n_features = 6
    features = np.zeros((n_channels, n_features, n_windows), dtype=np.float32)

    for i in range(n_windows):
        start_idx = i * hop_size
        end_idx = start_idx + window_size
        if end_idx > n_samples:
            break

        window_data = data[:, start_idx:end_idx]

        features[:, 0, i] = np.mean(window_data, axis=1)
        features[:, 1, i] = np.std(window_data, axis=1)
        features[:, 2, i] = np.min(window_data, axis=1)
        features[:, 3, i] = np.max(window_data, axis=1)
        features[:, 4, i] = skew(window_data, axis=1)
        features[:, 5, i] = kurtosis(window_data, axis=1)

    return features


def make_spectrogram(data, fs=50, nperseg=100, noverlap=50):
    """
    Create spectrogram from time-domain data
    Args:
        data: (time, channels) or (channels, time) array
        fs: sampling frequency
        nperseg: FFT window size
        noverlap: overlap size
    Returns:
        spec: (channels, freq_bins, time_bins) spectrogram
        freqs: frequency array
        times: time array
    """
    T, C = data.shape
    if C > 10:
        C, T = data.shape

    specs = []
    for ch in range(C):
        f, t, Sxx = spectrogram(
            data[:, ch] if T > C else data[ch, :],
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nperseg,
            scaling='density',
            mode='psd'
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        specs.append(Sxx_db[None, ...])

    spec = np.concatenate(specs, axis=0)
    return spec, f, t


def create_arousal_labels_extended(events, meas_date, total_samples, sfreq=50):
    """Create arousal labels from events"""
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

        if s_idx < 0:
            s_idx = 0
        if e_idx > total_samples:
            e_idx = total_samples

        if s_idx < total_samples:
            y[s_idx:e_idx] = 1.0

    return y


def map_label_to_spec_time(y_1d, t_array, fs=50, nperseg=100):
    """Map time-domain labels to spectrogram time bins"""
    half_win_sec = nperseg / (2.0 * fs)
    time_bins = len(t_array)
    label_spec_1d = np.zeros(time_bins, dtype=np.float32)

    for i, center_sec in enumerate(t_array):
        start_sec = center_sec - half_win_sec
        end_sec = center_sec + half_win_sec

        start_idx = int(np.floor(start_sec * fs))
        end_idx = int(np.ceil(end_sec * fs))

        if start_idx < 0:
            start_idx = 0
        if end_idx > len(y_1d):
            end_idx = len(y_1d)

        if np.any(y_1d[start_idx:end_idx] == 1):
            label_spec_1d[i] = 1.0
        else:
            label_spec_1d[i] = 0.0

    return label_spec_1d


def process_edf_arousal_multimodal(edf_path, xml_path, save_dir,
                                    fs=50,
                                    nperseg=100,
                                    noverlap=50,
                                    do_filter=False,
                                    extract_stat_features=False):
    """
    Process EDF file for multimodal arousal detection
    Saves both time-domain and frequency-domain features

    Args:
        edf_path: path to EDF file
        xml_path: path to arousal annotations XML
        save_dir: directory to save processed data
        fs: sampling frequency
        nperseg: spectrogram window size
        noverlap: spectrogram overlap
        do_filter: whether to apply bandpass filter
        extract_stat_features: whether to extract statistical features

    Returns:
        Dictionary with:
            - x_time: (channels, 6, time) time-domain features
            - x_spec: (channels, freq_bins, time_bins) spectrogram
            - y_time: (time,) time-domain labels
            - y_spec: (time_bins,) spectrogram labels
            - stat_features: (channels, 6, n_windows) if extract_stat_features=True
    """
    filename = basename(edf_path).replace(".edf", ".pkl")
    save_path = join(save_dir, filename)

    # Load EDF
    raw = load_edf_file(edf_path, preload=True, resample=fs, do_filter=do_filter)
    meas_date = raw.info['meas_date']
    data = raw.get_data()  # (channels, time)

    # Normalize data
    data_norm = robust_scale(data, fs=fs)

    # Load arousal events
    events = load_arousal_xml(xml_path)
    total_samples = data_norm.shape[1]
    y_time = create_arousal_labels_extended(events, meas_date, total_samples, sfreq=fs)

    # ==== TIME DOMAIN FEATURES ====
    # Extract amplitude-based features (channels, 6, time)
    x_time_features = extract_amplitude_features(data_norm, fs=fs)

    # Normalize each feature channel
    for ch in range(x_time_features.shape[0]):
        for feat in range(x_time_features.shape[1]):
            feat_data = x_time_features[ch, feat, :]
            x_time_features[ch, feat, :] = (feat_data - feat_data.mean()) / (feat_data.std() + 1e-8)

    # ==== FREQUENCY DOMAIN FEATURES ====
    # Create spectrogram
    data_transposed = data_norm.T  # (time, channels)
    spec, freqs, times = make_spectrogram(data_transposed, fs=fs,
                                          nperseg=nperseg,
                                          noverlap=noverlap)

    # Normalize spectrogram per channel
    spec_norm = np.zeros_like(spec)
    for ch in range(spec.shape[0]):
        spec_ch = spec[ch, :, :]
        spec_norm[ch, :, :] = (spec_ch - spec_ch.mean()) / (spec_ch.std() + 1e-8)

    # Map labels to spectrogram time bins
    y_spec = map_label_to_spec_time(y_time, times, fs=fs, nperseg=nperseg)

    # ==== OPTIONAL: STATISTICAL FEATURES ====
    stat_features = None
    if extract_stat_features:
        stat_features = extract_statistical_features(data_norm, fs=fs, window_sec=1.0)

    # ==== ARTIFACT DETECTION ====
    threshold = 1e-5
    per_ch_mask = np.all(spec < threshold, axis=1)
    artifact_mask = np.any(per_ch_mask, axis=0)

    print(f"[Artifact Mask] {np.sum(artifact_mask)}/{len(artifact_mask)} time bins")

    # ==== SAVE RESULT ====
    result = {
        "x_time": x_time_features.astype(np.float32),  # (C, 6, T)
        "x_spec": spec_norm.astype(np.float32),        # (C, F, T_spec)
        "y_time": y_time.astype(np.float32),           # (T,)
        "y_spec": y_spec.astype(np.float32),           # (T_spec,)
        "freqs": freqs,                                 # (F,)
        "times": times,                                 # (T_spec,)
        "artifact_mask": artifact_mask.astype(np.float32),
        "meas_date": meas_date,
        "fs": fs,
    }

    if stat_features is not None:
        result["stat_features"] = stat_features

    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[Saved Multimodal] {save_path}")
    print(f"  - Time features: {x_time_features.shape}")
    print(f"  - Spec features: {spec_norm.shape}")
    print(f"  - Time labels: {y_time.shape}, positives: {np.sum(y_time)}")
    print(f"  - Spec labels: {y_spec.shape}, positives: {np.sum(y_spec)}")

    return result


if __name__ == "__main__":
    sfreq = 50
    do_filter = False
    perseg = 1  # in seconds

    # Example dataset configuration
    base_dir = "/home/honeynaps/data/250718_CND"
    edf_dir = f"{base_dir}/EDF"
    xml_dir = f"{base_dir}/EBX/AROUS"

    tag = "multimodal_v1"

    # Save directory for multimodal features
    save_dir = f"{base_dir}/AROUS_MULTIMODAL/AROUSAL_MULTIMODAL_{sfreq}"
    if do_filter:
        save_dir += "_FILTERED"

    if len(tag) > 0:
        save_dir += f"_{tag}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]
    nperseg = perseg * sfreq
    overlap = 0.5

    swap = "_AROUS" if "HN_DATA" not in base_dir else "_ASHIFT"
    if "250428" in base_dir:
        swap = ""

    for i, edf_file in enumerate(edf_files):
        edf_path = os.path.join(edf_dir, edf_file)
        xml_path = os.path.join(xml_dir, edf_file.replace(".edf", f"{swap}.xml"))

        try:
            process_edf_arousal_multimodal(
                edf_path, xml_path, save_dir,
                fs=sfreq,
                nperseg=nperseg,
                noverlap=int(nperseg * overlap),
                do_filter=do_filter,
                extract_stat_features=False
            )
            print(f"Done processing {i+1}/{len(edf_files)}: {edf_file}\n")
        except Exception as e:
            print(f"Error: {edf_file}, {str(e)}")
            import traceback
            traceback.print_exc()
            continue
