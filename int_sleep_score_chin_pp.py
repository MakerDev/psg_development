import os
import sys
import argparse
import xml.etree.ElementTree as ET
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt
from dataclasses import dataclass

from arousal.utils.tools import load_edf_file
from arousal.ArousalFinal import ArousalFinal
from sleep_stage.SleepFinal import SleepFinal
from sleep_stage.modules.iofiles import edf as edf_io
from sleep_stage.utils.post_process import run_postprocess
from micro_event.models.crop_models import REDv2Time
from micro_event.datasets.dataset_hn_pred import SleepEventDatasetEBX
from micro_event.util.tools import save_micro_events_by_channels, save_micro_events_by_channels_and_type
from micro_event.postprocess.postprocessor import evaluate_edf, merge_and_prune, postprocess_preds
from tools.post_process_enhanced import correct_sleep_stages_with_tracking, PostProcessInfo
from tools.utils import str2bool, load_sleep_stage
from sklearn.metrics import confusion_matrix


# ======== Low Chin Tone Detection Components ========

def butter_highpass(cut_hz, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cut_hz/nyq, btype='highpass')
    return b, a


def highpass_filter(x, fs, cut_hz=5.0, order=4):
    if not cut_hz or cut_hz <= 0: 
        return x
    b, a = butter_highpass(cut_hz, fs, order)
    return filtfilt(b, a, x)


def rms_envelope(sig, fs, win_s=0.2, hop_s=0.1):
    win = max(1, int(round(win_s * fs)))
    hop = max(1, int(round(hop_s * fs)))
    sq = sig.astype(float) ** 2
    ker = np.ones(win) / float(win)
    mov = np.convolve(sq, ker, mode='same')
    env = np.sqrt(np.maximum(mov, 0.0))
    t = np.arange(len(env)) / fs
    return t[::hop], env[::hop]


def rolling_median_mad(x, win):
    s = pd.Series(x)
    med = s.rolling(win, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(win, center=True, min_periods=1).median()
    mad_scaled = 1.4826 * mad
    med = med.bfill().ffill()
    mad_scaled = mad_scaled.bfill().ffill()
    
    if (mad_scaled == 0).any():
        repl = float(np.median(mad_scaled[mad_scaled > 0])) if (mad_scaled > 0).any() else 1e-8
        mad_scaled = mad_scaled.replace(0, repl)
    
    return med.values, mad_scaled.values


@dataclass
class ChinDetectConfig:
    fs: int = 50
    hp_cut_hz: float = 5.0
    short_win_s: float = 0.2
    long_win_s: float = 1.0
    hop_s: float = 0.1
    baseline_win_s: float = 60.0
    z_start_short: float = 4.0
    z_end_short: float = 2.5
    z_start_long: float = 3.5
    z_end_long: float = 2.0
    r_start_short: float = 2.5
    r_end_short: float = 1.6
    r_start_long: float = 2.0
    r_end_long: float = 1.4
    min_event_dur_s: float = 0.10
    use_n3_threshold: bool = True
    n3_multiplier: float = 1.2


def _runs_from_bool(state):
    n = len(state)
    runs = []
    if n == 0:
        return runs
    
    cur = state[0]
    start = 0
    
    for i in range(1, n):
        if state[i] != cur:
            runs.append((cur, start, i))  # [start, i)
            cur = state[i]
            start = i
    
    runs.append((cur, start, n))
    return runs


def boolean_morph_cleanup(state_high, hop_s, remove_islands_s=0.3, fill_holes_s=0.3):
    """
    1) Remove short True islands (< remove_islands_s)
    2) Fill short False holes surrounded by True (< fill_holes_s)
    Returns cleaned boolean array.
    """
    st = state_high.copy()
    
    # Pass 1: remove short True islands
    runs = _runs_from_bool(st)
    for val, s, e in runs:
        dur = (e - s) * hop_s
        if val and dur < remove_islands_s:
            st[s:e] = False
    
    # Pass 2: fill short False holes surrounded by True
    runs = _runs_from_bool(st)
    for i, (val, s, e) in enumerate(runs):
        if not val:
            dur = (e - s) * hop_s
            if dur < fill_holes_s:
                left_true = (i - 1) >= 0 and runs[i - 1][0] == True
                right_true = (i + 1) < len(runs) and runs[i + 1][0] == True
                if left_true and right_true:
                    st[s:e] = True
    
    return st


def detect_chin_frame_state(raw_emg, cfg: ChinDetectConfig):
    fs = cfg.fs
    x = raw_emg.astype(float)
    x = x - np.median(x)
    
    if cfg.hp_cut_hz and cfg.hp_cut_hz > 0:
        x = highpass_filter(x, fs, cut_hz=cfg.hp_cut_hz, order=4)
    
    t_s, env_s = rms_envelope(x, fs, win_s=cfg.short_win_s, hop_s=cfg.hop_s)
    t_l, env_l = rms_envelope(x, fs, win_s=cfg.long_win_s, hop_s=cfg.hop_s)
    
    base_win = max(3, int(round(cfg.baseline_win_s / cfg.hop_s)))
    b_s, mad_s = rolling_median_mad(env_s, base_win)
    b_l, mad_l = rolling_median_mad(env_l, base_win)
    
    # Floors
    abs_floor = np.percentile(env_s, 60)
    mad_floor_s = np.percentile(mad_s, 30)
    mad_floor_l = np.percentile(mad_l, 30)
    b_floor_s = np.percentile(env_s, 10)
    b_floor_l = np.percentile(env_l, 10)
    
    mad_eff_s = np.maximum(mad_s, mad_floor_s)
    mad_eff_l = np.maximum(mad_l, mad_floor_l)
    b_eff_s = np.maximum(b_s, b_floor_s)
    b_eff_l = np.maximum(b_l, b_floor_l)
    
    z_s = (env_s - b_s) / (mad_eff_s + 1e-8)
    z_l = (env_l - b_l) / (mad_eff_l + 1e-8)
    r_s = env_s / np.maximum(b_eff_s, 1e-8)
    r_l = env_l / np.maximum(b_eff_l, 1e-8)
    
    amp_guard = (env_s > abs_floor) | (env_l > abs_floor)
    
    # Start/end conditions
    start = amp_guard & (
        ((z_s > cfg.z_start_short) & (r_s > cfg.r_start_short)) |
        ((z_l > cfg.z_start_long) & (r_l > cfg.r_start_long))
    )
    
    end = (
        (z_s < cfg.z_end_short) & (r_s < cfg.r_end_short) &
        (z_l < cfg.z_end_long) & (r_l < cfg.r_end_long)
    )
    
    state = np.zeros_like(env_s, dtype=bool)
    in_evt = False
    
    for i in range(len(env_s)):
        if not in_evt and start[i]:
            in_evt = True
        elif in_evt and end[i]:
            in_evt = False
        state[i] = in_evt
    
    # Morphological cleanup
    state_clean = boolean_morph_cleanup(state, cfg.hop_s,
                                       remove_islands_s=0.3,
                                       fill_holes_s=0.3)
    
    return {
        "t_short": t_s,
        "env_short": env_s,
        "base_short": b_s,
        "t_long": t_l,
        "env_long": env_l,
        "base_long": b_l,
        "state_high": state_clean
    }


def calculate_n3_threshold(sleep_stages, env_short, cfg: ChinDetectConfig):
    """Calculate N3-based absolute threshold"""
    n3_energies = []
    epoch_length_s = 30.0
    epoch_frames = int(epoch_length_s / cfg.hop_s)
    
    for epoch_idx, stage in enumerate(sleep_stages):
        if stage == 4:  # N3 stage
            start_frame = epoch_idx * epoch_frames
            end_frame = min((epoch_idx + 1) * epoch_frames, len(env_short))
            
            if start_frame < len(env_short):
                epoch_energies = env_short[start_frame:end_frame]
                n3_energies.extend(epoch_energies)
    
    if n3_energies:
        n3_average = np.mean(n3_energies)
        absolute_threshold = n3_average * cfg.n3_multiplier
        return absolute_threshold, n3_average
    else:
        return None, None


def apply_absolute_threshold(env_short, state_high, absolute_threshold):
    """Apply absolute threshold: if energy >= threshold, force to HIGH"""
    if absolute_threshold is None:
        return state_high
        
    state_modified = state_high.copy()
    high_energy_mask = env_short >= absolute_threshold
    state_modified[high_energy_mask] = True
    
    return state_modified


def merged_runs_from_state(t_frames, env_short, state_high, total_len_s,
                           min_dur_s=0.5, merge_same_gap_s=0.35):
    """Create merged HIGH/LOW runs"""
    hop = t_frames[1] - t_frames[0] if len(t_frames) > 1 else 0.1
    runs = _runs_from_bool(state_high)
    seq = []
    
    for val, s, e in runs:
        start = s * hop
        end = e * hop
        if end > total_len_s:
            end = total_len_s
        
        dur = end - start
        if dur < min_dur_s:
            continue
        
        avg = float(np.mean(env_short[s:e])) if e > s else 0.0
        seq.append({
            "start": start,
            "end": end,
            "duration": dur,
            "type": "HIGH" if val else "LOW",
            "avg_rms": avg
        })
    
    # Merge same-type runs separated by short gaps
    if not seq:
        return []
    
    merged = [seq[0]]
    for r in seq[1:]:
        prev = merged[-1]
        if r["type"] == prev["type"] and (r["start"] - prev["end"]) <= merge_same_gap_s:
            # Combine and recompute weighted avg_rms
            w1 = prev["duration"]
            w2 = r["duration"]
            avg = (prev["avg_rms"] * w1 + r["avg_rms"] * w2) / max(w1 + w2, 1e-8)
            prev["end"] = r["end"]
            prev["duration"] = prev["end"] - prev["start"]
            prev["avg_rms"] = avg
        else:
            merged.append(r)
    
    return merged


def pred_chin_tone(args):
    """
    Predict low chin tone events from EDF file.
    
    Args:
        args: Arguments containing EDF path, GPU settings, etc.
    
    Returns:
        chin_preds: List of chin tone events [(start_sec, duration_sec, type, avg_rms), ...]
        base_time: Recording start time
    """
    # Load EDF data
    edf, n_missing_ch = edf_io.load(
        path=args.edf,
        preload=True,
        resample=50,
        preset="STAGENET",
        exclude=True,
        missing_ch='raise'
    )
    
    base_time = edf.info['meas_date'].replace(tzinfo=None)
    if args.start_time is None:
        start_time = base_time
    else:
        start_time = datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=None)
    
    # Channel mapping
    SID_MAP = {
        'F3-': 'F3_2', 'F4-': 'F4_1', 'C3-': 'C3_2', 'C4-': 'C4_1',
        'O1-': 'O1_2', 'O2-': 'O2_1',
        'LOC': 'LOC', 'ROC': 'ROC',
        'EMG': 'CHIN'
    }
    
    data = edf.get_data()
    ch_names = edf.ch_names
    
    # Extract CHIN channel
    chin_idx = None
    for i, name in enumerate(ch_names):
        if name in SID_MAP and SID_MAP[name] == 'CHIN':
            chin_idx = i
            break
        elif name[:3] in SID_MAP and SID_MAP[name[:3]] == 'CHIN':
            chin_idx = i
            break
    
    if chin_idx is None:
        raise RuntimeError("CHIN channel not found in EDF file")
    
    # Get chin signal and convert to microvolts
    chin = data[chin_idx].astype(float) * 1000.0
    total_len_s = len(chin) / 50  # fs = 50
    
    # Initialize config
    cfg = ChinDetectConfig(fs=50)
    
    # Detect frame state
    dbg = detect_chin_frame_state(chin, cfg)
    t_frames = dbg["t_short"]
    env_short = dbg["env_short"]
    state_high = dbg["state_high"]
    
    # Optional: Apply N3-based absolute threshold
    if cfg.use_n3_threshold and hasattr(args, 'sleep_stages') and args.sleep_stages is not None:
        absolute_threshold, n3_average = calculate_n3_threshold(
            args.sleep_stages, env_short, cfg
        )
        if absolute_threshold is not None:
            print(f"Applying N3-based threshold: {absolute_threshold:.6f} (N3 avg: {n3_average:.6f})")
            state_high = apply_absolute_threshold(env_short, state_high, absolute_threshold)
    
    # Create merged runs
    runs = merged_runs_from_state(
        t_frames, env_short, state_high, total_len_s,
        min_dur_s=0.5, merge_same_gap_s=0.35
    )
    
    # Convert to prediction format
    chin_preds = []
    for run in runs:
        chin_preds.append((
            run["start"],           # start time in seconds
            run["duration"],        # duration in seconds
            run["type"],           # "HIGH" or "LOW"
            run["avg_rms"]         # average RMS value
        ))
    
    # Print summary
    n_high = sum(1 for r in runs if r["type"] == "HIGH")
    n_low = sum(1 for r in runs if r["type"] == "LOW")
    dur_high = sum(r["duration"] for r in runs if r["type"] == "HIGH")
    dur_low = sum(r["duration"] for r in runs if r["type"] == "LOW")
    
    print(f"Chin tone detection complete:")
    print(f"  HIGH chin tone: {n_high} events, {dur_high:.1f}s total")
    print(f"  LOW chin tone: {n_low} events, {dur_low:.1f}s total")
    
    return chin_preds, base_time


def save_chin_tone_xml(chin_preds, base_time, xml_path):
    """Save chin tone predictions to XML file"""
    root = ET.Element("annotationlist")
    
    # Add recording duration
    if chin_preds:
        last_event = chin_preds[-1]
        recording_duration = last_event[0] + last_event[1]
    else:
        recording_duration = 0
    
    ET.SubElement(root, "recording_duration").text = f"{recording_duration:.6f}"
    
    for start_sec, duration, event_type, avg_rms in chin_preds:
        onset_time = base_time + timedelta(seconds=start_sec)
        
        annotation = ET.SubElement(root, "annotation")
        
        onset_elem = ET.SubElement(annotation, "onset")
        onset_elem.text = onset_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
        
        duration_elem = ET.SubElement(annotation, "duration")
        duration_elem.text = f"{duration:.6f}"
        
        desc_elem = ET.SubElement(annotation, "description")
        desc_elem.text = f"{event_type}_CHIN_TONE_{avg_rms:.4f}"
        
        location_elem = ET.SubElement(annotation, "location")
        location_elem.text = "EEG-EMG"
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    
    with open(xml_path, "wb") as fp:
        fp.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(fp, encoding="UTF-8", xml_declaration=False)
    
    return len(chin_preds)


def calculate_low_chin_ratio_per_epoch(chin_preds, num_epochs, epoch_duration=30.0):
    """
    Calculate the ratio of low chin tone for each 30-second epoch.
    
    Args:
        chin_preds: List of chin tone events [(start_sec, duration_sec, type, avg_rms), ...]
        num_epochs: Total number of epochs in the recording
        epoch_duration: Duration of each epoch in seconds (default: 30.0)
    
    Returns:
        low_chin_ratios: Array of low chin tone ratios for each epoch
    """
    low_chin_ratios = np.zeros(num_epochs)
    
    for start_sec, duration_sec, event_type, avg_rms in chin_preds:
        if event_type == "LOW":
            # Calculate which epochs this event overlaps with
            start_epoch = int(start_sec / epoch_duration)
            end_sec = start_sec + duration_sec
            end_epoch = int(end_sec / epoch_duration)
            
            for epoch_idx in range(start_epoch, min(end_epoch + 1, num_epochs)):
                epoch_start = epoch_idx * epoch_duration
                epoch_end = (epoch_idx + 1) * epoch_duration
                
                # Calculate overlap duration
                overlap_start = max(start_sec, epoch_start)
                overlap_end = min(end_sec, epoch_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # Add to ratio
                low_chin_ratios[epoch_idx] += overlap_duration / epoch_duration
    
    # Ensure ratios don't exceed 1.0 (in case of overlapping events)
    low_chin_ratios = np.minimum(low_chin_ratios, 1.0)
    
    return low_chin_ratios


def correct_rem_with_chin_tone(sleep_preds, all_probs, low_chin_ratios, threshold=0.9):
    """
    Correct REM sleep predictions based on low chin tone ratio.
    
    Args:
        sleep_preds: Array of sleep stage predictions
        all_probs: Array of probabilities for each sleep stage (shape: [num_epochs, 5])
        low_chin_ratios: Array of low chin tone ratios for each epoch
        threshold: Minimum ratio of low chin tone required for REM (default: 0.9)
    
    Returns:
        corrected_preds: Corrected sleep stage predictions
        correction_stats: Dictionary with correction statistics
    """
    corrected_preds = sleep_preds.copy()
    
    # Track statistics
    correction_stats = {
        'total_rem_epochs': 0,
        'corrected_epochs': 0,
        'low_chin_insufficient': 0,
        'original_rem_probs': [],
        'next_best_probs': [],
        'next_best_stages': [],
        'stage_transitions': {0: 0, 1:0, 2: 0, 3: 0, 4: 0}  # Count transitions to each stage
    }
    
    for i in range(len(sleep_preds)):
        if sleep_preds[i] == 1:  # REM stage
            correction_stats['total_rem_epochs'] += 1
            
            # Check if low chin tone ratio is insufficient
            if low_chin_ratios[i] < threshold:
                correction_stats['low_chin_insufficient'] += 1
                
                # Get probabilities for this epoch
                epoch_probs = all_probs[i]
                
                # Find the second highest probability stage
                sorted_indices = np.argsort(epoch_probs)[::-1]  # Sort in descending order
                next_best_stage = sorted_indices[1]  # Second highest probability
                
                # Store statistics
                correction_stats['original_rem_probs'].append(epoch_probs[1])  # REM probability
                correction_stats['next_best_probs'].append(epoch_probs[next_best_stage])
                correction_stats['next_best_stages'].append(next_best_stage)
                correction_stats['stage_transitions'][next_best_stage] += 1
                
                # Apply correction
                corrected_preds[i] = next_best_stage
                correction_stats['corrected_epochs'] += 1
    
    # Calculate summary statistics
    if correction_stats['corrected_epochs'] > 0:
        correction_stats['avg_original_rem_prob'] = np.mean(correction_stats['original_rem_probs'])
        correction_stats['avg_next_best_prob'] = np.mean(correction_stats['next_best_probs'])
        correction_stats['std_original_rem_prob'] = np.std(correction_stats['original_rem_probs'])
        correction_stats['std_next_best_prob'] = np.std(correction_stats['next_best_probs'])
    else:
        correction_stats['avg_original_rem_prob'] = 0
        correction_stats['avg_next_best_prob'] = 0
        correction_stats['std_original_rem_prob'] = 0
        correction_stats['std_next_best_prob'] = 0
    
    return corrected_preds, correction_stats


def print_rem_correction_stats(correction_stats):
    """Print detailed statistics about REM corrections."""
    print("\n" + "="*60)
    print("REM CORRECTION STATISTICS (Low Chin Tone Analysis)")
    print("="*60)
    
    total_rem = correction_stats['total_rem_epochs']
    corrected = correction_stats['corrected_epochs']
    
    if total_rem > 0:
        correction_rate = (corrected / total_rem) * 100
        print(f"Total REM epochs: {total_rem}")
        print(f"Epochs with insufficient low chin tone (<90%): {correction_stats['low_chin_insufficient']}")
        print(f"Epochs corrected: {corrected} ({correction_rate:.1f}%)")
        
        if corrected > 0:
            print(f"\nProbability Analysis:")
            print(f"  Average REM probability (corrected epochs): {correction_stats['avg_original_rem_prob']:.3f} ± {correction_stats['std_original_rem_prob']:.3f}")
            print(f"  Average next-best stage probability: {correction_stats['avg_next_best_prob']:.3f} ± {correction_stats['std_next_best_prob']:.3f}")
            
            print(f"\nStage Transitions from REM:")
            stage_names = {0: 'Wake', 1: 'REM', 2: 'N1', 3: 'N2', 4: 'N3'}
            for stage, count in correction_stats['stage_transitions'].items():
                if count > 0:
                    percentage = (count / corrected) * 100
                    print(f"  REM → {stage_names[stage]}: {count} epochs ({percentage:.1f}%)")
    else:
        print("No REM epochs found in the recording.")
    
    print("="*60 + "\n")


# ======== Original Functions from int_sleep_score2.py ========

def save_enhanced_sleepstage_xml(meas_date, original_stages, corrected_stages, 
                                post_process_info, ground_truth, xml_save_path, 
                                location="EEG-F4"):
    """Save sleep stage predictions with detailed post-processing analysis to XML."""
    label_to_stage = {
        0: "SLEEP-W",
        1: "SLEEP-R", 
        2: "SLEEP-1",
        3: "SLEEP-2",
        4: "SLEEP-3"
    }

    root = ET.Element("annotationlist")

    for i, (original_stage, corrected_stage, gt_stage) in enumerate(zip(original_stages, corrected_stages, ground_truth)):
        start_sec = i * 30
        onset_time = meas_date + timedelta(seconds=start_sec)
        duration = 30.0
        
        pp_info = post_process_info[i] if i < len(post_process_info) else None
        
        stage_name = label_to_stage.get(corrected_stage, "SLEEP-U")
        is_correct = "TRUE" if corrected_stage == gt_stage else "FALSE"
        
        post_process_wrong = "FALSE"
        post_process_reason = "NONE"
        
        if pp_info and pp_info.was_changed:
            if original_stage == gt_stage and corrected_stage != gt_stage:
                post_process_wrong = "TRUE"
            post_process_reason = pp_info.change_reason
        
        description = f"{stage_name}_{is_correct}_{post_process_wrong}_{post_process_reason}"

        annotation = ET.SubElement(root, "annotation")
        onset_elem = ET.SubElement(annotation, "onset")
        onset_elem.text = onset_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
        duration_elem = ET.SubElement(annotation, "duration")
        duration_elem.text = f"{duration:.6f}"
        desc_elem = ET.SubElement(annotation, "description")
        desc_elem.text = description
        location_elem = ET.SubElement(annotation, "location")
        location_elem.text = location

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(xml_save_path, encoding="UTF-8", xml_declaration=True)


def save_to_xml(preds, save_path, base_time, description, location):
    root = ET.Element("annotationlist")

    for pe in preds:
        onset_time = base_time + timedelta(seconds=pe[0])
        
        annotation = ET.SubElement(root, "annotation")
        onset_elem = ET.SubElement(annotation, "onset")
        onset_elem.text = onset_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

        duration_elem = ET.SubElement(annotation, "duration")
        duration_elem.text = f"{pe[1]:.6f}"

        desc_elem = ET.SubElement(annotation, "description")
        desc_elem.text = description

        location_elem = ET.SubElement(annotation, "location")
        location_elem.text = location

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(save_path, encoding="UTF-8", xml_declaration=True)


def evaluate_model(model, val_loader, device, th=0.1557, max_len_sec=5, min_len_sec=0.5):
    model.eval()
    all_probs = {}

    with torch.no_grad():
        for X, info in val_loader:        
            X = X.to(device)

            logits = model(X)                       
            if logits.ndim > 2:
                logits = logits.squeeze(1)          

            probs = torch.softmax(logits, dim=-1)[..., 1]

            batch_size = X.size(0)
            channel_names = info[1] if isinstance(info, (list, tuple)) else ['default'] * batch_size

            for b in range(batch_size):
                ch_name = channel_names[b]
                all_probs.setdefault(ch_name, []).append(probs[b].cpu())


    for ch_name in all_probs:
        all_probs[ch_name] = torch.cat(all_probs[ch_name], dim=0).numpy()

    all_preds = {}
    for ch_name in all_probs:
        all_probs[ch_name] = all_probs[ch_name].reshape(-1)
        all_preds[ch_name] = (all_probs[ch_name] > th).astype(int)
        all_preds[ch_name] = merge_and_prune(all_preds[ch_name], fs=200//8, 
                                            max_len_sec=max_len_sec,
                                            min_len_sec=min_len_sec,
                                            merge_th=0.1)
    
    return all_preds


def pred_arousal(args):
    edf = load_edf_file(
        path=args.edf, 
        preload=True, 
        resample=50, 
        preset="STAGENET", 
        exclude=True,
        missing_ch='raise'
    )
    
    base_time = edf.info['meas_date'].replace(tzinfo=None)
    if args.start_time is None:
        start_time = base_time
    else:
        start_time = datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=None)
    
    SID_MAP = { 
        'F3-': 'F3_2', 'F4-': 'F4_1', 'C3-': 'C3_2', 'C4-': 'C4_1', 
        'O1-': 'O1_2', 'O2-': 'O2_1', 
        'LOC': 'LOC', 'ROC': 'ROC', 
        'EMG': 'CHIN'
    }
    data = edf.get_data()

    sigs = {}
    for i in range(len(edf.ch_names)):
        name = edf.ch_names[i]
        if name in SID_MAP:
            sigs[SID_MAP[name]] = data[i]
        else:
            sigs[SID_MAP[name[:3]]] = data[i]
    
    detector = ArousalFinal(
        sigs, base_time, 
        start_time=start_time,
        gpu=args.gpu,
        seed=args.seed,
        num_channels=args.num_channels,
        fs=args.fs,
        type=args.type,
        ver=args.ver,
        tag=args.tag
    )

    pretrained_dir = "/home/honeynaps/data/shared/arousal/saved_models"
    preds = detector(pretrained_dir)

    return preds, base_time


def pred_sleep_stage(args):
    if not (args.edf or args.dest):
        print('Arguments "--edf" or "--dest" required!!!')
        os._exit(1)

    edf, n_missing_ch = edf_io.load(
        path=args.edf, 
        preload=True, 
        resample=50, 
        preset="STAGENET", 
        exclude=True,
        missing_ch='raise'
    )
    
    base_time = edf.info['meas_date'].replace(tzinfo=None)
    if args.start_time is None:
        start_time = base_time
    else:
        start_time = datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=None)
    
    SID_MAP = { 
        'F3-': 'F3_2', 'F4-': 'F4_1', 'C3-': 'C3_2', 'C4-': 'C4_1', 
        'O1-': 'O1_2', 'O2-': 'O2_1', 
        'LOC': 'LOC', 'ROC': 'ROC', 
        'EMG': 'CHIN'
    }
    data = edf.get_data()

    sigs = {}
    for i in range(len(edf.ch_names)):
        name = edf.ch_names[i]
        if name in SID_MAP:
            sigs[SID_MAP[name]] = data[i]
        else:
            sigs[SID_MAP[name[:3]]] = data[i]

    detector = SleepFinal(
        sigs, base_time, 
        start_time=start_time,
        model='resnet18',
        gpu=args.gpu,
        seed=42,
        num_channels=9,
        fs=50,
        nofill=True,
        tag=args.tag
    )

    pretrained_dir = '/home/honeynaps/data/shared/sleep_stage/saved_models'
    y_pred, all_probs = detector(pretrained_dir)

    return y_pred, all_probs  # Return both predictions and probabilities


def pred_micro_event(args, th_mul=1.3):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    file_names = [args.edf]

    if args.start_time:
        start_time = datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=None)
    else:
        start_time = None

    sleep_dataset = SleepEventDatasetEBX(
        file_names,
        start_times=[start_time],
        page_duration=args.page_duration
    )
    data_loader = torch.utils.data.DataLoader(
        sleep_dataset,
        batch_size=32,
        shuffle=False, 
        num_workers=4
    )

    save_dir = "/home/honeynaps/data/shared/micro_event/saved_models"

    model = REDv2Time(in_channels=1)
    model.to(device)

    if args.event_type == 'kcomplex':
        pretrained_path = f'{save_dir}/HN_kcomplex_ep012_f10.4473_newall_th0.2433.pth'
        th, max_len_sec, min_len_sec = 0.1557, 5, 0.5
    elif args.event_type == 'spindle':
        pretrained_path = f'{save_dir}/HN_spindle_ep006_f10.5243_newall_th0.2657.pth'
        th, max_len_sec, min_len_sec = 0.2725, 3, 0.5
    
    th = th * th_mul
    
    if args.pretrained and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))

    preds_all = evaluate_model(model, data_loader, device, th=th,
                              max_len_sec=max_len_sec, min_len_sec=min_len_sec)

    preds_all = postprocess_preds(preds_all, sleep_dataset,
                                 event_type=args.event_type,
                                 page_duration=args.page_duration)
    
    return preds_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str, default="/home/honeynaps/data/GOLDEN/EDF2/SCH_F_40_NW_230511R3_SE.edf")
    parser.add_argument('--dest', type=str, default="/home/honeynaps/data/shared/integrate")
    parser.add_argument('--start_time', type=str, default=None, help='Start time in format "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ver', type=int, default=2)
    parser.add_argument('--num_channels', type=int, default=9)
    parser.add_argument('--fs', type=int, default=50)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--type', type=str, default='spec', choices=['time', 'spec', 'union', 'intersection'])
    parser.add_argument('--seed', type=int, default=0)

    # Micro Event Parameters
    parser.add_argument('--page_duration', type=int, default=10)  # seconds
    parser.add_argument('--event_type', type=str, default='kcomplex', choices=['kcomplex', 'spindle'])
    parser.add_argument('--pretrained', type=str2bool, default=True)
    parser.add_argument('--th_mul', type=float, default=1.3, help='Threshold multiplier for micro event detection')
    
    # Chin Tone Parameters
    parser.add_argument('--detect_chin', type=str2bool, default=True, help='Enable chin tone detection')
    parser.add_argument('--use_n3_threshold', type=str2bool, default=True, help='Use N3-based absolute threshold')
    
    # REM Correction Parameters
    parser.add_argument('--rem_chin_threshold', type=float, default=0.9, help='Minimum low chin tone ratio for REM (default: 0.9)')

    args = parser.parse_args()

    if not (args.edf or args.dest):
        print('Arguments "--edf" or "--dest" required!!!')
        os._exit(1)

    # Extract EDF filename without extension
    edf_basename = os.path.splitext(os.path.basename(args.edf))[0]
    
    # Load ground truth labels
    sleep_label_xml_path = args.edf.replace('.edf', '_SLEEP.xml').replace('EDF2', 'EBX2/SLEEP').replace('EDF', 'EBX/SLEEP')
    sleep_labels, start_time = load_sleep_stage(sleep_label_xml_path)

    args.start_time = start_time

    # Run predictions
    print("Running arousal prediction...")
    arousal_preds, base_time = pred_arousal(args)
    
    print("Running sleep stage prediction...")
    sleep_preds, all_probs = pred_sleep_stage(args)  # Get both predictions and probabilities
    original_sleep_preds = sleep_preds.copy()
    
    print("Running micro event prediction...")
    micro_event_preds_by_channels = pred_micro_event(args)
    
    # Run chin tone detection
    if args.detect_chin:
        print("Running chin tone detection...")
        # Pass sleep stages for N3 threshold calculation if enabled
        if args.use_n3_threshold:
            args.sleep_stages = sleep_preds
        chin_preds, chin_base_time = pred_chin_tone(args)
        
        # Save chin tone predictions to XML
        chin_xml_filename = f"{edf_basename}_CHIN_TONE.xml"
        chin_xml_path = os.path.join(args.dest, chin_xml_filename)
        n_chin_events = save_chin_tone_xml(chin_preds, chin_base_time, chin_xml_path)
        print(f"Saved {n_chin_events} chin tone events to: {chin_xml_path}")
        
        # Calculate low chin tone ratio per epoch
        num_epochs = len(sleep_preds)
        low_chin_ratios = calculate_low_chin_ratio_per_epoch(chin_preds, num_epochs)
        
        # Apply REM correction based on low chin tone
        print("\nApplying REM correction based on low chin tone ratio...")
        sleep_preds_after_chin, rem_correction_stats = correct_rem_with_chin_tone(
            sleep_preds, all_probs, low_chin_ratios, threshold=args.rem_chin_threshold
        )
        
        # Print REM correction statistics
        print_rem_correction_stats(rem_correction_stats)
        
        # Update sleep predictions with chin-based corrections
        sleep_preds = sleep_preds_after_chin
    
    # Align lengths
    min_length = min(len(sleep_preds), len(sleep_labels))
    sleep_preds = sleep_preds[:min_length]
    original_sleep_preds = original_sleep_preds[:min_length]
    sleep_labels = sleep_labels[:min_length]
    
    print("Sleep Accuracy before post-processing correction:")
    n_corrected, n_total = 0, 0
    for i in range(len(sleep_preds)):
        if sleep_preds[i] == sleep_labels[i]:
            n_corrected += 1
        n_total += 1
    print(f"Corrected: {n_corrected}, Total: {n_total}, Accuracy: {n_corrected / n_total:.2f}")
    cm = confusion_matrix(sleep_labels, sleep_preds)
    print("Confusion Matrix:")
    print(cm)

    # Apply post-processing correction with tracking
    print("\nApplying micro-event based post-processing correction...")
    corrected_sleep_stages, post_process_info = correct_sleep_stages_with_tracking(
        arousal_preds, 
        sleep_preds, 
        micro_event_preds_by_channels
    )

    print("\nSleep Accuracy after all corrections:")
    n_corrected, n_total = 0, 0
    for i in range(len(corrected_sleep_stages)):
        if corrected_sleep_stages[i] == sleep_labels[i]:
            n_corrected += 1
        n_total += 1
    print(f"Corrected: {n_corrected}, Total: {n_total}, Accuracy: {n_corrected / n_total:.2f}")
    cm = confusion_matrix(sleep_labels, corrected_sleep_stages)
    print("Confusion Matrix:")
    print(cm)

    # Analyze post-processing impact
    changes_count = sum(1 for info in post_process_info if info.was_changed)
    correct_to_wrong = sum(1 for i, info in enumerate(post_process_info) 
                          if info.was_changed and sleep_preds[i] == sleep_labels[i] and corrected_sleep_stages[i] != sleep_labels[i])
    wrong_to_correct = sum(1 for i, info in enumerate(post_process_info) 
                          if info.was_changed and sleep_preds[i] != sleep_labels[i] and corrected_sleep_stages[i] == sleep_labels[i])
    
    print(f"\nMicro-Event Post-processing Analysis:")
    print(f"Total epochs changed: {changes_count}")
    print(f"Correct -> Wrong: {correct_to_wrong}")
    print(f"Wrong -> Correct: {wrong_to_correct}")
    print(f"Net improvement: {wrong_to_correct - correct_to_wrong}")

    # Save enhanced XML with post-processing analysis
    xml_filename = f"{edf_basename}_SLEEP_ANALYSIS_CHIN_PP.xml"
    xml_save_path = os.path.join(args.dest, xml_filename)
    
    print(f"\nSaving enhanced XML to: {xml_save_path}")
    save_enhanced_sleepstage_xml(
        base_time, 
        original_sleep_preds, 
        corrected_sleep_stages,
        post_process_info,
        sleep_labels, 
        xml_save_path
    )

    print("\nAnalysis completed!")