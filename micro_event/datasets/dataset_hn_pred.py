# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pyedflib
import torch
import sys
sys.path.append("/home/honeynaps/data/eis/SEED_pytorch")
from util import utils

channel_map: Dict[str, int] = {
    "F3-M2": 0,
    "F4-M1": 1,
    "C3-M2": 2,
    "C4-M1": 3,
    "O1-M2": 4,
    "O2-M1": 5,
}

channel_rename_map: Dict[str, str] = {
    "C3-A2": "C3-M2",
    "C4-A1": "C4-M1",
    "O1-A2": "O1-M2",
    "O2-A1": "O2-M1",
    "F3-A2": "F3-M2",
    "F4-A1": "F4-M1",
}

def _match_channel(edf_labels: List[str], target: str) -> int:
    """Return EDF channel index that matches *target* (with or without "EEG " prefix)."""
    variants = {target, f"EEG {target}", target.replace("-", " - ")}  # be tolerant
    for idx, lbl in enumerate(edf_labels):
        if lbl.strip() in variants:
            return idx
    raise ValueError(f"Channel {target} not found in EDF labels: {edf_labels}")


class SleepEventDatasetEBX(torch.utils.data.Dataset):
    """PyTorch `Dataset` for the EBX EEG corpus (without event labels)."""

    def _fix_signal_length(self, signal: np.ndarray) -> np.ndarray:
        """Trim signal to multiple of page_size."""
        n_pages = len(signal) // self.page_size
        n_valid = n_pages * self.page_size
        return signal[:n_valid]

    # ------------------------------------------------------------------
    # Construction helpers (low‑level I/O)
    # ------------------------------------------------------------------

    def _read_eeg_signal(self, edf_path: Path, channel_name: str) -> np.ndarray:
        """Return **broad‑band filtered**, resampled (→ ``self.fs``) EEG trace."""
        with pyedflib.EdfReader(str(edf_path)) as edf:
            raw_labels = edf.getSignalLabels()
            mapped_labels = [channel_rename_map.get(lbl, lbl) for lbl in raw_labels]
            ch_idx = _match_channel(mapped_labels, channel_name)
            raw_sig = edf.readSignal(ch_idx)
            fs_old = edf.samplefrequency(ch_idx)
            rec_start_time = edf.getStartdatetime().replace(tzinfo=None)

        raw_sig = utils.resample_signal(raw_sig, 500, self.fs)
        raw_sig = utils.broad_filter(raw_sig, self.fs)

        return raw_sig.astype(np.float32), rec_start_time

    def _load_subject_channel(
        self,
        sid: str,
        channel_name: str,
        edf_path: Path,
        start_time: Optional[dt.datetime] = None,
    ) -> Dict[str, np.ndarray]:
        signals, rec_start_time = self._read_eeg_signal(edf_path, channel_name)
        
        # If start_time is provided, skip samples before it
        if start_time is not None:
            skip_seconds = (start_time - rec_start_time).total_seconds()
            if skip_seconds > 0:
                skip_samples = int(skip_seconds * self.fs)
                signals = signals[skip_samples:]
            self.base_time = start_time
        else:
            self.base_time = rec_start_time

        signals = self._fix_signal_length(signals)
        n_pages = len(signals) // self.page_size
        all_pages = np.arange(n_pages, dtype=np.int16)
        
        return {
            "sid": sid,
            "channel": channel_name,
            "signal": signals,
            "all_pages": all_pages,
        }
    
    def get_start_time(self):
        return self.base_time

    def __init__(
        self,
        edf_paths: List[str],
        *,
        page_duration: int = 10,
        target_fs: int = 200,
        augmented_page: bool = False,
        border_sec: float = 2.6,
        normalize_clip: bool = True,
        stride: int = 8,
        start_times: Optional[List[dt.datetime]] = None,
    ) -> None:
        super().__init__()
        self.fs                 = target_fs
        self.page_duration      = page_duration                 
        self.page_size          = self.fs * self.page_duration  
        self.augmented_page     = augmented_page                
        self.border_size        = int(round(border_sec * self.fs))
        self.normalize_clip     = normalize_clip
        self.stride             = stride
        self.start_times        = start_times or {}

        # Storage for *channel‑specific* entries
        self.entries: List[Dict[str, np.ndarray]] = []

        for idx, edf_path in enumerate(edf_paths):
            sid = os.path.basename(edf_path).split('.')[0] 
            start_time = self.start_times[idx] if self.start_times else None

            for ch in channel_map.keys():
                entry = self._load_subject_channel(sid, ch, edf_path, start_time)
                self.entries.append(entry)

        # Stats+tensors
        self.global_std = self._calculate_global_std()
        print(f"Global std: {self.global_std:.2f} uV")
        
        self.signals, self.user_infos, self.raw_signals = self._prepare_data()

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int):
        feat = self.signals[idx]
        user_info = self.user_infos[idx]
        
        # 1. random crop to (page + borders)
        tot_len = feat.shape[-1]
        crop_len = self.page_size + 2 * self.border_size
        if tot_len > crop_len:
            start = np.random.randint(0, tot_len - crop_len + 1)
            end = start + crop_len
            feat = feat[start:end]

        # 2. → tensor
        feat_t = torch.from_numpy(feat).float().unsqueeze(0)  # (1, L)
        return feat_t, user_info

    def _calculate_global_std(self) -> float:
        total, s1, s2 = 0, 0.0, 0.0
        for e in self.entries:
            x = e["signal"].copy()
            median_val = np.median(x)
            mad = np.median(np.abs(x - median_val))
            x = (x - median_val) / (1.4826 * mad)
            x *= 10
            e["signal"] = x  # update entry with normalized signal

            thr = np.percentile(np.abs(x), 99)
            if thr < 1:
                x = x * 1e3
                e["signal"] = x

            # Use all pages for std calculation
            pages = e["all_pages"]
            x = utils.extract_pages(x, pages, self.page_size).flatten()
            thr = np.percentile(np.abs(x), 99)
            mean_x = x[np.abs(x) <= thr].mean()

            x = x[np.abs(x) <= thr]

            total += x.size
            s1 += x.sum()
            s2 += (x ** 2).sum()
            
        mean_sq = s2 / total
        mean = s1 / total
        return float(np.sqrt(mean_sq - mean ** 2))

    def _get_processed_entry(
        self,
        entry: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        signals = entry["signal"].copy()
        pages = entry["all_pages"].astype(int)

        # Normalisation (global clipping)
        if self.normalize_clip:
            # Use all pages for normalization
            signals, _ = utils.norm_clip_signal(
                signals,
                pages,
                self.page_size,
                norm_computation="global",
                global_std=self.global_std,
                clip_value=10,
            )

        # Extract all pages (+optional borders)
        total_border = (
            self.page_size // 2 + self.border_size if self.augmented_page else self.border_size
        )
        signals = utils.extract_pages(signals, pages, self.page_size, total_border)
        raw_signals = utils.extract_pages(
            entry["signal"].copy(), pages, self.page_size, total_border
        )

        return signals.astype(np.float32), raw_signals.astype(np.float32)

    def _prepare_data(self):
        signals, user_infos, raw_signals = [], [], []
        for entry in self.entries:
            signal, raw_signal = self._get_processed_entry(entry)
            signals.append(signal)
            raw_signals.append(raw_signal)
            user_infos.extend([(entry["sid"], entry["channel"])] * len(signal))

        signals = np.concatenate(signals, axis=0)
        raw_signals = np.concatenate(raw_signals, axis=0)
        
        return signals, user_infos, raw_signals


# -----------------------------------------------------------------------------
# Quick test (disable for production – kept for illustration)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    root = "/home/honeynaps/data/HN_DATA_MW"  # ← Adjust
    subjects = os.listdir(root + "/" + "EDF2")
    subjects = [s.split(".")[0] for s in subjects if s.endswith(".edf")]
    
    # Example: provide start times for each subject (if needed)
    start_times = {}  # You can populate this with actual start times
    
    ds = SleepEventDatasetEBX(
        root, 
        subjects, 
        page_duration=20,
        start_times=start_times
    )
    print(f"Dataset size: {len(ds)}")
    
    x, user_info = ds[0]
    print(f"Signal shape: {x.shape}")
    print(f"User info: {user_info}")