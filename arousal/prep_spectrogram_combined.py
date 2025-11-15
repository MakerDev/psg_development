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
from scipy.signal import spectrogram, hilbert, butter, filtfilt
from scipy.stats import skew, kurtosis

import random
from utils.tools import *


# ==============================
# Normalization Functions
# ==============================
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


# ==============================
# Time Domain Feature Extraction
# ==============================
def extract_amplitude_envelope(data, fs=50):
    """
    Extract amplitude envelope using Hilbert transform
    data: (channels, time)
    returns: (channels, time)
    """
    envelope = np.zeros_like(data, dtype=np.float32)

    for ch_idx in range(data.shape[0]):
        analytic_signal = hilbert(data[ch_idx])
        envelope[ch_idx] = np.abs(analytic_signal)

    return envelope


def extract_amplitude_derivatives(envelope, fs=50):
    """
    Extract first and second derivatives of amplitude envelope
    envelope: (channels, time)
    returns: first_deriv (channels, time), second_deriv (channels, time)
    """
    first_deriv = np.gradient(envelope, axis=1) * fs
    second_deriv = np.gradient(first_deriv, axis=1) * fs

    return first_deriv, second_deriv


def extract_bandpower_features(data, fs=50, window_sec=2):
    """
    Extract band power features in sliding windows
    data: (channels, time)
    returns: (channels, n_bands, time_windows)
    """
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
    }

    window_size = int(window_sec * fs)
    stride = window_size // 2  # 50% overlap

    n_channels = data.shape[0]
    n_time = data.shape[1]
    n_windows = (n_time - window_size) // stride + 1
    n_bands = len(bands)

    bandpower_features = np.zeros((n_channels, n_bands, n_windows), dtype=np.float32)

    for ch_idx in range(n_channels):
        for win_idx in range(n_windows):
            start_idx = win_idx * stride
            end_idx = start_idx + window_size

            if end_idx > n_time:
                break

            segment = data[ch_idx, start_idx:end_idx]

            # Compute PSD
            f, psd = spectrogram(segment, fs=fs, nperseg=min(256, window_size),
                                noverlap=min(128, window_size//2))

            # Extract band power
            for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
                freq_mask = (f >= low) & (f <= high)
                bandpower_features[ch_idx, band_idx, win_idx] = np.sum(psd[freq_mask])

    return bandpower_features


def extract_statistical_features(data, fs=50, window_sec=2):
    """
    Extract statistical features in sliding windows
    data: (channels, time)
    returns: (channels, n_features, time_windows)
    """
    window_size = int(window_sec * fs)
    stride = window_size // 2

    n_channels = data.shape[0]
    n_time = data.shape[1]
    n_windows = (n_time - window_size) // stride + 1
    n_features = 6  # mean, std, min, max, skewness, kurtosis

    stat_features = np.zeros((n_channels, n_features, n_windows), dtype=np.float32)

    for ch_idx in range(n_channels):
        for win_idx in range(n_windows):
            start_idx = win_idx * stride
            end_idx = start_idx + window_size

            if end_idx > n_time:
                break

            segment = data[ch_idx, start_idx:end_idx]

            stat_features[ch_idx, 0, win_idx] = np.mean(segment)
            stat_features[ch_idx, 1, win_idx] = np.std(segment)
            stat_features[ch_idx, 2, win_idx] = np.min(segment)
            stat_features[ch_idx, 3, win_idx] = np.max(segment)
            stat_features[ch_idx, 4, win_idx] = skew(segment)
            stat_features[ch_idx, 5, win_idx] = kurtosis(segment)

    return stat_features


# ==============================
# Frequency Domain Feature Extraction
# ==============================
def make_spectrogram(data, fs=50, nperseg=100, noverlap=50):
    """
    Create spectrogram for multi-channel data
    data: (time, channels)
    returns: spec (channels, freq_bins, time_bins), freqs, times
    """
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
        Sxx_db = 10 * np.log1p(Sxx + 1e-12)
        specs.append(Sxx_db[None, ...])

    spec = np.concatenate(specs, axis=0)
    return spec, f, t


# ==============================
# Label Creation
# ==============================
def create_arousal_labels_extended(events, meas_date, total_samples, sfreq=50):
    """Create binary arousal labels"""
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


# ==============================
# Combined Feature Extraction and Chunking
# ==============================
def process_edf_arousal_combined(edf_path, xml_path, save_dir,
                                 chunk_duration=60,
                                 fs=50,
                                 nperseg=100,
                                 noverlap=50,
                                 do_filter=False):
    """
    Process EDF file and extract multimodal features with chunking

    Args:
        edf_path: Path to EDF file
        xml_path: Path to arousal annotation XML
        save_dir: Directory to save processed data
        chunk_duration: Duration of each chunk in seconds (default: 60)
        fs: Sampling frequency
        nperseg: FFT window size for spectrogram
        noverlap: Overlap for spectrogram
        do_filter: Whether to apply bandpass filter
    """
    filename = basename(edf_path).replace(".edf", "")

    # Load EDF file
    raw = load_edf_file(edf_path, preload=True, resample=fs, do_filter=do_filter)
    meas_date = raw.info['meas_date']
    data = raw.get_data()  # (channel, time)

    # Normalize raw data
    data_norm = robust_scale(data, fs=fs)

    # Load arousal labels
    events = load_arousal_xml(xml_path)
    total_samples = data_norm.shape[1]
    y = create_arousal_labels_extended(events, meas_date, total_samples, sfreq=fs)

    # Process in chunks
    chunk_samples = int(chunk_duration * fs)
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    print(f"Processing {filename}: {n_chunks} chunks of {chunk_duration}s each")

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_samples
        end_idx = min((chunk_idx + 1) * chunk_samples, total_samples)

        # Extract chunk
        data_chunk = data_norm[:, start_idx:end_idx]  # (channels, time)
        y_chunk = y[start_idx:end_idx]

        # Skip if chunk is too short
        if data_chunk.shape[1] < fs * 10:  # At least 10 seconds
            print(f"Skipping chunk {chunk_idx}: too short")
            continue

        # ===== Time Domain Features =====
        # 1. Raw normalized signal
        x_time_raw = data_chunk.T  # (time, channels)

        # 2. Amplitude envelope
        envelope = extract_amplitude_envelope(data_chunk, fs=fs)  # (channels, time)

        # 3. Amplitude derivatives
        first_deriv, second_deriv = extract_amplitude_derivatives(envelope, fs=fs)

        # 4. Combine time features: raw + envelope + derivatives
        x_time_combined = np.stack([
            data_chunk,      # (channels, time)
            envelope,        # (channels, time)
            first_deriv,     # (channels, time)
            second_deriv     # (channels, time)
        ], axis=1)  # (channels, 4, time)

        # ===== Frequency Domain Features =====
        spec, freqs, times = make_spectrogram(x_time_raw, fs=fs,
                                             nperseg=nperseg,
                                             noverlap=noverlap)

        # Normalize spectrogram
        spec = (spec - spec.mean(axis=(1, 2), keepdims=True)) / \
               (spec.std(axis=(1, 2), keepdims=True) + 1e-8)

        # ===== Amplitude Features =====
        # Statistical features
        stat_features = extract_statistical_features(data_chunk, fs=fs, window_sec=2)

        # ===== Labels =====
        # Map labels to spectrogram time
        label_spec = map_label_to_spec_time(y_chunk, times, fs=fs, nperseg=nperseg)

        # ===== Artifact Detection =====
        threshold = 1e-5
        per_ch_mask = np.all(spec < threshold, axis=1)
        artifact_mask = np.any(per_ch_mask, axis=0)

        # ===== Save chunk =====
        chunk_filename = f"{filename}_chunk{chunk_idx:04d}.pkl"
        save_path = join(save_dir, chunk_filename)

        result = {
            # Time domain features
            "x_time_raw": x_time_raw.astype(np.float32),           # (time, channels)
            "x_time_combined": x_time_combined.astype(np.float32), # (channels, 4, time)
            "envelope": envelope.astype(np.float32),                # (channels, time)

            # Frequency domain features
            "x_spec": spec.astype(np.float32),                     # (channels, freq, time_bins)
            "freqs": freqs,
            "times": times,

            # Amplitude features
            "stat_features": stat_features.astype(np.float32),     # (channels, 6, time_windows)

            # Labels
            "y_time": y_chunk.astype(np.float32),                  # (time,)
            "y_spec": label_spec.astype(np.float32),               # (time_bins,)

            # Metadata
            "artifact_mask": artifact_mask.astype(np.float32),
            "meas_date": meas_date,
            "chunk_idx": chunk_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "fs": fs,
        }

        with open(save_path, "wb") as f:
            pickle.dump(result, f)

        print(f"  Saved chunk {chunk_idx}/{n_chunks}: {chunk_filename}")

    print(f"[Completed] {filename}: {n_chunks} chunks saved")
    return n_chunks


# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    sfreq = 50
    chunk_duration = 60  # 60 seconds per chunk

    # Dataset paths - modify as needed
    base_dir = "/home/honeynaps/data/250718_CND"
    edf_dir = f"{base_dir}/EDF"
    xml_dir = f"{base_dir}/EBX/AROUS"

    tag = "multimodal_60s"
    do_filter = False
    perseg = 1

    save_dir = f"{base_dir}/AROUS_MULTIMODAL/AROUSAL_COMBINED_{sfreq}"

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
            n_chunks = process_edf_arousal_combined(
                edf_path, xml_path, save_dir,
                chunk_duration=chunk_duration,
                fs=sfreq,
                nperseg=nperseg,
                noverlap=int(nperseg * overlap),
                do_filter=do_filter
            )
            print(f"Done processing {i+1}/{len(edf_files)}: {edf_file} ({n_chunks} chunks)")
        except Exception as e:
            print(f"Error: {edf_file}, {str(e)}")
            import traceback
            traceback.print_exc()
            continue
