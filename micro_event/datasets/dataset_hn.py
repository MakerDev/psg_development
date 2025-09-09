# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import datetime as dt
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyedflib
import torch
import sys
sys.path.append("/home/honeynaps/data/eis/SEED_pytorch")
from util import utils

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
channel_map_train: Dict[str, int] = {
    "F3-M2": 0, # 1966 11860 (spindle)
    "F4-M1": 1, # 1966 11983 (spindle)
    "C3-M2": 2, # 2020 11910 (spindle)
    "C4-M1": 3, # 1961 11925 (spindle)
    "O1-M2": 4, # 1977 12011 (spindle)
    "O2-M1": 5, # 1959 11992 (spindle)
}

channel_map_test: Dict[str, int] = {
    "F3-M2": 0, # 2200 11700 (kcomplex)
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

sleep_stage_map: Dict[str, str] = {
    "SLEEP-WAKE": "W",
    "SLEEP-REM": "R",
    "SLEEP-N1": "1",
    "SLEEP-N2": "2",
    "SLEEP-N3": "3",  # treat N3 as stage 3; merge N3/4 if desired later
}

# Reverse map used when we need numeric ids (e.g. mask creation)
stage_id_map: Dict[str, int] = {v: i for i, v in enumerate(["W", "R", "1", "2", "3"])}
unknown_id = "?"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _parse_iso_datetime(ts: str) -> dt.datetime:
    """Convert ISO 8601 string with micro‑seconds into **naïve** datetime."""
    return dt.datetime.fromisoformat(ts.replace("Z", ""))


def _match_channel(edf_labels: List[str], target: str) -> int:
    """Return EDF channel index that matches *target* (with or without "EEG " prefix)."""
    variants = {target, f"EEG {target}", target.replace("-", " - ")}  # be tolerant
    for idx, lbl in enumerate(edf_labels):
        if lbl.strip() in variants:
            return idx
    raise ValueError(f"Channel {target} not found in EDF labels: {edf_labels}")


# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------

class SleepEventDatasetEBX(torch.utils.data.Dataset):
    """PyTorch `Dataset` for the EBX EEG corpus (single‑channel sleep micro‑events)."""

    def _fix_signal_and_states(
        self, signal: np.ndarray, hypnogram: np.ndarray, start_sample: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        # (identical to original implementation – only docstring removed for brevity)
        signal = signal[start_sample:]
        n_sig = (signal.size // self.page_size) * self.page_size
        n_hyp = hypnogram.size * self.page_size
        n_valid = min(n_sig, n_hyp)
        n_pages = n_valid // self.page_size
        return signal[:n_valid], hypnogram[:n_pages], start_sample + n_valid

    def _hypnogram_selections(self, hypnogram: np.ndarray):
        total = len(hypnogram)
        n2_pages = np.where(hypnogram == "2")[0].astype(np.int16)
        n2_pages = n2_pages[(n2_pages != 0) & (n2_pages != total - 1)]
        all_pages = np.arange(1, total - 1, dtype=np.int16)
        return all_pages, n2_pages
    
    def _fix_marks(
        self, marks: np.ndarray, start_sample: int, end_sample: int
    ) -> np.ndarray:
        marks = marks - start_sample
        end_sample -= start_sample
        return utils.filter_stamps(marks, 0, end_sample - 1)

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

        # Round frequency (e.g. 199.999 → 200)
        # fs_old_round = int(round(fs_old))
        # raw_sig = utils.resample_signal_linear(raw_sig, fs_old, fs_old_round)
        original_fs = fs_old
        raw_sig = utils.resample_signal(raw_sig, 500, self.fs)
        raw_sig = utils.broad_filter(raw_sig, self.fs, highcut=10)
        return raw_sig.astype(np.float32)

    # def _remap_sleep_stages(self, page_onsets, page_stages, target_duration=20):
    #     remapped_onsets = []
    #     remapped_stages = []

    #     current_onset = page_onsets[0]
    #     current_stage = page_stages[0]

    #     for i in range(len(page_onsets)):
    #         onset = page_onsets[i]
    #         stage = page_stages[i]

    #         while current_onset + 30 < onset:
    #             # Fill in gaps between annotations with the dominant stage
    #             remapped_onsets.append(current_onset)
    #             remapped_stages.append(current_stage)
    #             current_onset += target_duration  # Move forward by 20 seconds

    #         if onset - current_onset >= 20:
    #             # If the gap is significant, randomly assign stages within the gap
    #             num_segments = int((onset - current_onset) / target_duration)
    #             for _ in range(num_segments):
    #                 remapped_onsets.append(current_onset)
    #                 remapped_stages.append(np.random.choice([current_stage, stage], p=[0.5, 0.5]))
    #                 current_onset += target_duration

    #         # Assign the main stage for the current 20-second segment
    #         remapped_onsets.append(current_onset)
    #         remapped_stages.append(stage)
    #         current_onset += target_duration

    #     # Fill any remaining gaps with the last observed stage
    #     while current_onset < page_onsets[-1] + 30:
    #         remapped_onsets.append(current_onset)
    #         remapped_stages.append(current_stage)
    #         current_onset += target_duration

    #     return np.array(remapped_onsets), np.array(remapped_stages)



    def _remap_sleep_stages(self, page_onsets, page_stages, target_duration=20):
        page_onsets = np.asarray(page_onsets, dtype=float)
        page_stages = np.asarray(page_stages)

        if page_onsets.ndim != 1 or page_stages.ndim != 1:
            raise ValueError("page_onsets and page_stages must be 1‑D")
        if len(page_onsets) != len(page_stages):
            raise ValueError("page_onsets and page_stages must have the same length")
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")

        order = np.argsort(page_onsets)
        page_onsets = page_onsets[order]
        page_stages = page_stages[order]

        start_time = page_onsets[0]
        end_time   = page_onsets[-1] + target_duration
        remap_onsets = np.arange(start_time, end_time, target_duration, dtype=float)

        idx = np.searchsorted(page_onsets, remap_onsets, side='right') - 1
        idx[idx < 0] = 0

        remap_stages = page_stages[idx]

        return remap_onsets, remap_stages

    def _read_hypnogram(self, xml_path: Path) -> Tuple[np.ndarray, int]:
        """Parse *SLEEP.xml* → page‑wise stage vector + absolute start sample."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        rec_start = _parse_iso_datetime(root.findtext("recording_start_time"))
        page_onsets, page_stages = [], []
        for annot in root.iter("annotation"):
            label = annot.findtext("description")
            if label not in sleep_stage_map:
                continue  # ignore unknown labels
            onset_ts = _parse_iso_datetime(annot.findtext("onset"))
            dur = float(annot.findtext("duration"))  # should be 30.0
            rel_sec = (onset_ts - rec_start).total_seconds()
            page_onsets.append(rel_sec)
            page_stages.append(sleep_stage_map[label])

        page_onsets = np.asarray(page_onsets)
        order = np.argsort(page_onsets)
        page_onsets = page_onsets[order]
        page_stages = np.asarray(page_stages)[order]

        if self.page_duration < 30:
            page_onsets, page_stages = self._remap_sleep_stages(page_onsets, page_stages, self.page_duration)
            # print(f"Remapped sleep stages to {self.page_duration}s intervals.")

        # The EDF file may start before the first annotated page
        start_time = page_onsets[0]
        start_sample = int(start_time * self.fs)

        onsets_pages = np.round(page_onsets / self.page_duration).astype(int)
        n_pages = 1 + onsets_pages[-1]
        hypnogram = np.full(n_pages, unknown_id, dtype="U1")
        hypnogram[onsets_pages] = page_stages
        return hypnogram, start_sample

    def get_start_time(self, subject_idx: str) -> dt.datetime:
        """Return the start time of the recording for a given subject."""
        xml_path = self.root_dir / "EBX" / "SLEEP" / f"{self.subject_ids[subject_idx]}_SLEEP.xml"

        tree = ET.parse(xml_path)
        root = tree.getroot()

        rec_start = _parse_iso_datetime(root.findtext("recording_start_time"))
        page_onsets = []
        for annot in root.iter("annotation"):
            label = annot.findtext("description")
            if label not in sleep_stage_map:
                continue  # ignore unknown labels
            onset_ts = _parse_iso_datetime(annot.findtext("onset"))
            dur = float(annot.findtext("duration"))  # should be 30.0
            rel_sec = (onset_ts - rec_start).total_seconds()
            page_onsets.append((rel_sec, onset_ts))

        # sort by rel sec
        order = np.argsort([onset[0] for onset in page_onsets])
        page_onset_ts = np.array([onset[1] for onset in page_onsets])[order]
        # order = np.argsort(page_onsets)
        # page_onsets = page_onsets[order]
        start_time = page_onset_ts[0]

        return rec_start, start_time


    def _read_events(self, xml_path: Path, channel_name: str, expand_sec=0.0) -> np.ndarray:
        """Return *sample‑wise* onset/offset indices (N×2) for desired event type."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        rec_start = _parse_iso_datetime(root.findtext("recording_start_time"))

        desired_code = "MW_EEG-KCOMP" if self.event_type == "kcomplex" else "MW_EEG-SPIND"
        marks = []
        for annot in root.iter("annotation"):
            if annot.findtext("description") != desired_code:
                continue
            if annot.findtext("location") != f"EEG-{channel_name}":
                continue

            onset_ts = _parse_iso_datetime(annot.findtext("onset"))
            dur = float(annot.findtext("duration"))
            rel_sec = (onset_ts - rec_start).total_seconds()
            start, end = rel_sec * self.fs, (rel_sec + dur) * self.fs
            expand = int(round(expand_sec * self.fs))
            marks.append([int(round(start)) - expand, int(round(end)) + expand])

        if not marks:
            return np.empty((0, 2), dtype=int)
        return np.asarray(marks, dtype=int)

    # ..................................................................

    def _load_subject_channel(
        self,
        sid: str,
        channel_name: str,
        edf_path: Path,
        sleep_xml: Path,
        event_xml: Path,
    ) -> Dict[str, np.ndarray]:
        signals = self._read_eeg_signal(edf_path, channel_name)
        hypno, start_sample = self._read_hypnogram(sleep_xml)
        signals, hypno, end_sample = self._fix_signal_and_states(signals, hypno, start_sample)
        all_pages, n2_pages = self._hypnogram_selections(hypno)
        marks = self._read_events(event_xml, channel_name, expand_sec=self.expand_sec)
        marks = self._fix_marks(marks, start_sample, end_sample)
        if np.any(marks > end_sample):
            msg = "Values in intervals should be within end bound"
            # drop last mark if it exceeds end_sample
            marks = marks[marks[:, 0] < end_sample]
            marks = marks[marks[:, 1] <= end_sample]
            if len(marks) == 0:
                msg = "No valid marks found within end bound"
                raise ValueError(msg)

        if np.any(marks > len(signals)):
            msg = "Values in intervals should be within signal length"
            # drop last mark if it exceeds signal length
            marks = marks[marks[:, 0] < len(signals)]
            marks = marks[marks[:, 1] <= len(signals)]
            if len(marks) == 0:
                msg = "No valid marks found within signal length"
                raise ValueError(msg)

        return {
            "sid": sid,
            "channel": channel_name,
            "signal": signals,
            "hypnogram": hypno,
            "n2_pages": n2_pages,
            "all_pages": all_pages,
            "marks": marks,
        }


    def __init__(
        self,
        root_dir: str | Path,
        subject_ids: List[str],
        *,
        event_type: str = "spindle",  # "spindle" | "kcomplex"
        page_duration: int = 30,
        target_fs: int = 200,
        augmented_page: bool = False,
        border_sec: float = 2.6,
        normalize_clip: bool = True,
        pages_subset: str = "N2",
        stride: int = 8,
        expand_sec: float = 0.0,  # seconds to expand event marks
    ) -> None:
        super().__init__()
        self.root_dir           = Path(root_dir)
        self.subject_ids        = subject_ids
        self.event_type         = event_type.lower()
        self.fs                 = target_fs
        self.page_duration      = page_duration                 # seconds
        self.page_size          = self.fs * self.page_duration  # samples 
        self.augmented_page     = augmented_page                # True if we want to add borders around the page
        self.border_size        = int(round(border_sec * self.fs))
        self.normalize_clip     = normalize_clip
        self.pages_subset       = pages_subset                  # "N2" | "all" (use all pages)
        self.stride             = stride
        self.expand_sec         = expand_sec  # seconds to expand event marks

        # Storage for *channel‑specific* entries
        self.entries: List[Dict[str, np.ndarray]] = []
        global channel_map_train
        global channel_map_test

        if not augmented_page:
            channel_map_train = channel_map_test

        for sid in subject_ids:
            edf_path  = self.root_dir / "EDF2" / f"{sid}.edf"
            sleep_xml = self.root_dir / "EBX" / "SLEEP" / f"{sid}_SLEEP.xml"
            # event_xml = self.root_dir / "EBX" / "MW_EEG" / f"{sid}_MW_EEG.xml"
            event_xml = self.root_dir / "EBX" / "MW_EEG_NEW_ALL" / f"{sid}_MW_EEG.xml"

            for ch in channel_map_train.keys():
                entry = self._load_subject_channel(sid, ch, edf_path, sleep_xml, event_xml)
                self.entries.append(entry)

        # Stats+tensors
        self.global_std  = self._calculate_global_std()
        self.channel_std = self._calculate_channel_std()

        print(f"Global std: {self.global_std:.2f} uV")
        self.signals, self.marks, self.page_masks = self._prepare_data()

        seg_has_event = self.marks.sum(axis=1) > 0
        self.pos_segments = int(seg_has_event.sum())
        self.neg_segments = int(self.marks.shape[0] - self.pos_segments)
        print(self.pos_segments, self.neg_segments)

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int):
        feat, label, mask = (
            self.signals[idx],
            self.marks[idx],
            self.page_masks[idx],
        )
        # 1. random crop to (page + borders)
        tot_len = feat.shape[-1]
        crop_len = self.page_size + 2 * self.border_size
        if tot_len > crop_len:
            start = np.random.randint(0, tot_len - crop_len + 1)
            end = start + crop_len
            feat, label, mask = feat[start:end], label[start:end], mask[start:end]

        # 2. centre region (page only)
        center_label = label[self.border_size : -self.border_size]
        center_mask  = mask[self.border_size : -self.border_size]

        # 3. downsample (aligned mean‑pool + rounding)
        blk  = self.stride
        trim = (len(center_label) // blk) * blk
        label_blocks = center_label[:trim].reshape(-1, blk)
        mask_blocks  = center_mask[:trim].reshape(-1, blk)
        
        label_down = np.rint(label_blocks.mean(axis=1)).astype(np.float32)
        mask_down  = np.rint(mask_blocks.mean(axis=1)).astype(np.float32)

        # 4. → tensors
        feat_t  = torch.from_numpy(feat).float().unsqueeze(0)  # (1, L)
        label_t = torch.from_numpy(label_down).float()
        mask_t  = torch.from_numpy(mask_down).float()
        return feat_t, label_t, mask_t

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

            hypno = e["hypnogram"]
            pages = np.concatenate([np.where(hypno == lbl)[0] for lbl in ["1", "2", "3", "R"]])
            x = utils.extract_pages(x, pages, self.page_size).flatten()
            thr = np.percentile(np.abs(x), 99)
            mean_x = x[np.abs(x) <= thr].mean()

            # print(f"Signal {e['sid']} ({e['channel']}): mean({mean_x}) clipping threshold = {thr:.2f} uV")
            x = x[np.abs(x) <= thr]

            total += x.size
            s1 += x.sum()
            s2 += (x ** 2).sum()
        mean_sq = s2 / total
        mean = s1 / total
        return float(np.sqrt(mean_sq - mean ** 2))

    def _calculate_channel_std(self) -> Dict[str, float]:
        stats: Dict[str, Dict[str, float]] = {
            ch: {"total": 0, "s1": 0.0, "s2": 0.0} for ch in channel_map_train
        }

        for e in self.entries:
            ch_name = e["channel"]
            x = e["signal"].copy()

            # 1) 중앙값 기반 de‑trend & MAD 정규화(기존 로직 유지)
            median_val = np.median(x)
            mad = np.median(np.abs(x - median_val))
            x = (x - median_val) / (1.4826 * mad)
            x *= 10
            e["signal"] = x  # entry 업데이트

            # 2) 99‑percentile 클리핑 전 σ 추정용 샘플 집계
            thr = np.percentile(np.abs(x), 99)
            if thr < 1:                      # very small amplitude ⇒ μV 단위로 재조정
                x = x * 1e3
                e["signal"] = x

            hypno = e["hypnogram"]
            pages = np.concatenate(
                [np.where(hypno == lbl)[0] for lbl in ["1", "2", "3", "R"]]
            )
            seg = utils.extract_pages(x, pages, self.page_size).flatten()
            thr = np.percentile(np.abs(seg), 99)
            seg = seg[np.abs(seg) <= thr]    # robust 구간만 사용

            n = seg.size
            stats[ch_name]["total"] += n
            stats[ch_name]["s1"]   += seg.sum()
            stats[ch_name]["s2"]   += (seg ** 2).sum()

        channel_std = {}
        for ch, s in stats.items():
            mean_sq = s["s2"] / s["total"]
            mean    = s["s1"] / s["total"]
            channel_std[ch] = float(np.sqrt(mean_sq - mean ** 2))
            print(f"[STD] {ch}: {channel_std[ch]:.2f} µV")

        return channel_std

    # ..................................................................

    def _get_processed_entry(
        self,
        entry: Dict[str, np.ndarray],
        *,
        pages_subset: str = "N2",
        forced_mark_separation_size: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        signals = entry["signal"].copy()
        marks   = entry["marks"].copy()
        if forced_mark_separation_size:
            marks = utils.stamp2seq_with_separation(
                marks, 0, signals.shape[0] - 1, forced_mark_separation_size
            )
        else:
            marks = utils.stamp2seq(marks, 0, signals.shape[0] - 1)

        pages = (entry["n2_pages"] if pages_subset == "N2" else entry["all_pages"]).astype(int)
        page_mask = utils.stamp2seq(
            np.stack([pages * self.page_size, (pages + 1) * self.page_size - 1], axis=1),
            0,
            signals.shape[0] - 1,
        )

        # Normalisation (global clipping)
        if self.normalize_clip:
            std = self.channel_std.get(entry["channel"], self.global_std)
            signals, _ = utils.norm_clip_signal(
                signals,
                entry["n2_pages"],
                self.page_size,
                norm_computation="global",
                global_std=std,
                clip_value=10,
            )

        # Extract selected pages (+optional borders)
        total_border = (
            self.page_size // 2 + self.border_size if self.augmented_page else self.border_size
        )
        signals   = utils.extract_pages(signals, pages, self.page_size, total_border)
        marks     = utils.extract_pages(marks, pages, self.page_size, total_border)
        page_mask = utils.extract_pages(page_mask, pages, self.page_size, total_border)

        # Drop pages with no events with probability of 0.2 for each page
        # no_event_pages = np.where(marks.sum(axis=1) == 0)[0]
        # if len(no_event_pages) > 0:
        #     keep_mask = np.random.rand(len(no_event_pages)) > 0.2
        #     keep_mask = np.concatenate([np.ones(len(marks) - len(no_event_pages), dtype=bool), keep_mask])
        #     signals = signals[keep_mask]
        #     marks = marks[keep_mask]
        #     page_mask = page_mask[keep_mask]


        return signals.astype(np.float32), marks.astype(np.int8), page_mask.astype(np.int8)

    # ..................................................................

    def _prepare_data(self):
        signals, marks, page_masks = [], [], []
        for entry in self.entries:
            signal, mark, page_mask = self._get_processed_entry(entry, pages_subset=self.pages_subset)
            signals.append(signal)
            marks.append(mark)
            page_masks.append(page_mask)

        signals = np.concatenate(signals, axis=0)
        marks = np.concatenate(marks, axis=0)
        page_masks = np.concatenate(page_masks, axis=0)

        return signals, marks, page_masks



# -----------------------------------------------------------------------------
# Quick test (disable for production – kept for illustration)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    root = "/home/honeynaps/data/HN_DATA_MW"  # ← Adjust
    subjects = os.listdir(root + "/" + "EDF2")
    subjects = [s.split(".")[0] for s in subjects if s.endswith(".edf")]
    ds = SleepEventDatasetEBX(root, subjects, page_duration=20, event_type="spindle")
    print(len(ds))
    x, y, m = ds[0]
    print(x.shape, y.shape, m.shape)
