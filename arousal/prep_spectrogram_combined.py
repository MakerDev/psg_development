import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import pickle
from os.path import basename, join

import numpy as np
from scipy.signal import spectrogram, hilbert

from utils.tools import load_edf_file, load_arousal_xml, create_arousal_labels


def robust_scale(x: np.ndarray) -> np.ndarray:
    median = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - median), axis=1, keepdims=True) + 1e-9
    return (x - median) / mad


def make_spectrogram(data: np.ndarray, fs: int, nperseg: int, noverlap: int):
    n_channels = data.shape[0]
    specs = []
    for ch in range(n_channels):
        f, t, Sxx = spectrogram(
            data[ch],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nperseg,
            scaling="density",
            mode="psd",
        )
        Sxx = 10 * np.log10(Sxx + 1e-12)
        specs.append(Sxx[None, ...])
    spec = np.concatenate(specs, axis=0)
    return spec.astype(np.float32), f, t


def compute_time_features(data: np.ndarray, fs: int, times: np.ndarray, window_size: int):
    n_channels, total_samples = data.shape
    half_window = window_size // 2
    n_windows = len(times)

    feature_names = [
        "mean",
        "std",
        "abs_mean",
        "line_length",
        "energy",
        "hilbert_mean",
        "hilbert_max",
        "slope",
    ]

    features = np.zeros((n_channels, len(feature_names), n_windows), dtype=np.float32)

    for t_idx, center_time in enumerate(times):
        center_sample = int(round(center_time * fs))
        start = max(center_sample - half_window, 0)
        end = start + window_size
        if end > total_samples:
            end = total_samples
            start = max(end - window_size, 0)

        segment = data[:, start:end]
        if segment.shape[1] < window_size:
            pad = window_size - segment.shape[1]
            segment = np.pad(segment, ((0, 0), (0, pad)), mode="edge")

        abs_seg = np.abs(segment)
        diff_seg = np.diff(segment, axis=1)
        envelope = np.abs(hilbert(segment, axis=1))

        features[:, 0, t_idx] = np.mean(segment, axis=1)
        features[:, 1, t_idx] = np.std(segment, axis=1)
        features[:, 2, t_idx] = np.mean(abs_seg, axis=1)
        features[:, 3, t_idx] = np.mean(np.abs(diff_seg), axis=1)
        features[:, 4, t_idx] = np.mean(segment ** 2, axis=1)
        features[:, 5, t_idx] = np.mean(envelope, axis=1)
        features[:, 6, t_idx] = np.max(envelope, axis=1)
        features[:, 7, t_idx] = segment[:, -1] - segment[:, 0]

    # Abrupt change descriptors based on first-order differences
    change_features = []
    change_names = []
    for idx, name in enumerate(["abs_mean", "energy", "hilbert_mean"]):
        series = features[:, feature_names.index(name), :]
        delta = np.diff(series, prepend=series[:, :1], axis=1)
        change_features.append(delta[:, None, :])
        change_names.append(f"d_{name}")

    if change_features:
        change_matrix = np.concatenate(change_features, axis=1)
        features = np.concatenate([features, change_matrix], axis=1)
        feature_names.extend(change_names)

    # Normalise per-channel for stability
    mean = np.mean(features, axis=2, keepdims=True)
    std = np.std(features, axis=2, keepdims=True)
    std[std < 1e-6] = 1e-6
    features = (features - mean) / std

    return features.astype(np.float32), feature_names


def map_label_to_spec_time(labels: np.ndarray, times: np.ndarray, fs: int, window_size: int):
    half_window_sec = window_size / (2.0 * fs)
    mapped = np.zeros(len(times), dtype=np.float32)

    for idx, center_time in enumerate(times):
        start_sec = center_time - half_window_sec
        end_sec = center_time + half_window_sec
        start = int(np.floor(start_sec * fs))
        end = int(np.ceil(end_sec * fs))
        start = max(start, 0)
        end = min(end, len(labels))
        if np.any(labels[start:end] == 1):
            mapped[idx] = 1.0
    return mapped


def detect_artifact_mask(spec: np.ndarray, threshold: float = 1e-5):
    per_channel_mask = np.all(spec < threshold, axis=1)
    return np.any(per_channel_mask, axis=0).astype(np.uint8)


def process_record(
    edf_path: str,
    xml_path: str,
    save_dir: str,
    fs: int,
    nperseg: int,
    noverlap: int,
    do_filter: bool = False,
):
    filename = basename(edf_path).replace(".edf", ".pkl")
    save_path = join(save_dir, filename)

    raw = load_edf_file(edf_path, preload=True, resample=fs, do_filter=do_filter)
    meas_date = raw.info["meas_date"]
    data = raw.get_data()

    data_norm = robust_scale(data)
    events = load_arousal_xml(xml_path)
    total_samples = data_norm.shape[1]
    labels = create_arousal_labels(events, meas_date, total_samples, sfreq=fs)

    spec, freqs, times = make_spectrogram(data_norm, fs=fs, nperseg=nperseg, noverlap=noverlap)
    spec = (spec - np.mean(spec, axis=(1, 2), keepdims=True)) / (np.std(spec, axis=(1, 2), keepdims=True) + 1e-6)

    time_features, feature_names = compute_time_features(data_norm, fs=fs, times=times, window_size=nperseg)
    artifact_mask = detect_artifact_mask(spec)

    mapped_labels = map_label_to_spec_time(labels, times, fs=fs, window_size=nperseg)

    result = {
        "spectrogram": spec.astype(np.float32),
        "time_features": time_features.astype(np.float32),
        "feature_names": feature_names,
        "freqs": freqs.astype(np.float32),
        "times": times.astype(np.float32),
        "y": mapped_labels.astype(np.float32),
        "artifact_mask": artifact_mask,
    }

    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Create combined spectrogram + time-domain features")
    parser.add_argument("edf_dir", type=str, help="Directory with EDF files")
    parser.add_argument("xml_dir", type=str, help="Directory with XML annotation files")
    parser.add_argument("save_dir", type=str, help="Directory to save pickle files")
    parser.add_argument("--sfreq", type=int, default=50)
    parser.add_argument("--nperseg", type=int, default=100)
    parser.add_argument("--noverlap", type=int, default=50)
    parser.add_argument("--limit", type=int, default=-1, help="Process only first N files")
    parser.add_argument("--do_filter", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    edf_files = [f for f in sorted(os.listdir(args.edf_dir)) if f.lower().endswith(".edf")]
    if args.limit > 0:
        edf_files = edf_files[:args.limit]

    for idx, edf_file in enumerate(edf_files):
        edf_path = join(args.edf_dir, edf_file)
        xml_name = edf_file.replace(".edf", "_AROUS.xml")
        xml_path = join(args.xml_dir, xml_name)
        if not os.path.exists(xml_path):
            print(f"[Skip] Missing XML for {edf_file}")
            continue
        try:
            output_path = process_record(
                edf_path,
                xml_path,
                args.save_dir,
                fs=args.sfreq,
                nperseg=args.nperseg,
                noverlap=args.noverlap,
                do_filter=args.do_filter,
            )
            print(f"[{idx + 1}/{len(edf_files)}] Saved {output_path}")
        except Exception as exc:
            print(f"[Error] {edf_file}: {exc}")


if __name__ == "__main__":
    main()
