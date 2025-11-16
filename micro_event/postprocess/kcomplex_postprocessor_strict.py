"""
Strict K-Complex Post-processing with Clinical Standards

This module enforces strict K-complex criteria:
1. Minimum amplitude: 75 µV (clinical standard)
2. Clear biphasic waveform with good shape quality
3. Baseline context analysis (quiet surrounding activity)
4. Peak prominence and symmetry validation
"""

import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences, peak_widths


def find_zero_crossings(signal, fs=200):
    """Find zero-crossing points in a signal"""
    sign = np.sign(signal)
    sign[sign == 0] = -1
    zero_crossings = np.where(np.diff(sign))[0]
    return zero_crossings


def calculate_baseline_noise(signal, event_start, event_end, fs=200, window_sec=1.0):
    """
    Calculate baseline noise level before and after the event

    Args:
        signal: full signal
        event_start, event_end: event boundaries
        fs: sampling frequency
        window_sec: window size for baseline calculation

    Returns:
        baseline_std: standard deviation of baseline
        baseline_mean: mean of baseline
    """
    window_samples = int(window_sec * fs)

    # Before event
    before_start = max(0, event_start - window_samples)
    before_segment = signal[before_start:event_start]

    # After event
    after_end = min(len(signal), event_end + window_samples)
    after_segment = signal[event_end:after_end]

    # Combine baseline segments
    if len(before_segment) > 0 and len(after_segment) > 0:
        baseline = np.concatenate([before_segment, after_segment])
    elif len(before_segment) > 0:
        baseline = before_segment
    elif len(after_segment) > 0:
        baseline = after_segment
    else:
        return np.nan, np.nan

    return np.std(baseline), np.mean(baseline)


def calculate_shape_quality(signal, pos_peak_idx, neg_peak_idx, fs=200):
    """
    Calculate shape quality metrics for K-complex

    Returns:
        dict with quality metrics:
            - peak_ratio: ratio of positive to negative peak prominences (should be ~1)
            - symmetry_score: symmetry of the waveform
            - sharpness_score: how sharp/distinct the peaks are
            - overall_quality: combined quality score (0-1)
    """
    # Calculate peak prominences
    pos_prom = peak_prominences(signal, [pos_peak_idx])[0][0]
    neg_prom = peak_prominences(-signal, [neg_peak_idx])[0][0]

    # 1. Peak ratio (should be balanced, ideally 0.5-2.0)
    peak_ratio = pos_prom / (neg_prom + 1e-8)
    peak_ratio_score = 1.0 if 0.5 <= peak_ratio <= 2.0 else max(0, 1 - abs(np.log2(peak_ratio)))

    # 2. Symmetry (biphasic waveform should be relatively symmetric)
    # Calculate widths at half prominence
    try:
        pos_width = peak_widths(signal, [pos_peak_idx], rel_height=0.5)[0][0]
        neg_width = peak_widths(-signal, [neg_peak_idx], rel_height=0.5)[0][0]

        width_ratio = pos_width / (neg_width + 1e-8)
        symmetry_score = 1.0 if 0.5 <= width_ratio <= 2.0 else max(0, 1 - abs(np.log2(width_ratio)))
    except:
        symmetry_score = 0.5

    # 3. Sharpness (peaks should be distinct, not rounded)
    # Calculate slope at peak positions
    window = int(0.05 * fs)  # 50ms window

    # Positive peak sharpness
    pos_start = max(0, pos_peak_idx - window)
    pos_end = min(len(signal), pos_peak_idx + window)
    pos_slopes = np.abs(np.diff(signal[pos_start:pos_end]))
    pos_sharpness = np.mean(pos_slopes)

    # Negative peak sharpness
    neg_start = max(0, neg_peak_idx - window)
    neg_end = min(len(signal), neg_peak_idx + window)
    neg_slopes = np.abs(np.diff(signal[neg_start:neg_end]))
    neg_sharpness = np.mean(neg_slopes)

    # Normalize sharpness (higher is better)
    avg_signal_std = np.std(signal)
    sharpness_score = min(1.0, (pos_sharpness + neg_sharpness) / (2 * avg_signal_std + 1e-8))

    # 4. Overall quality (weighted combination)
    overall_quality = (
        0.4 * peak_ratio_score +
        0.3 * symmetry_score +
        0.3 * sharpness_score
    )

    return {
        'peak_ratio': peak_ratio,
        'peak_ratio_score': peak_ratio_score,
        'symmetry_score': symmetry_score,
        'sharpness_score': sharpness_score,
        'overall_quality': overall_quality,
        'pos_prominence': pos_prom,
        'neg_prominence': neg_prom
    }


def detect_kcomplex_peaks_strict(signal, fs=200, min_prominence_ratio=0.15):
    """
    Strict peak detection for K-complex with shape quality validation

    Args:
        signal: 1D array (should be filtered 0.5-4 Hz)
        fs: sampling frequency
        min_prominence_ratio: minimum peak prominence relative to signal std

    Returns:
        dict with validation results
    """
    result = {
        'has_valid_pattern': False,
        'pos_peak_idx': None,
        'neg_peak_idx': None,
        'pos_amplitude': 0,
        'neg_amplitude': 0,
        'peak_to_peak': 0,
        'duration': 0,
        'shape_quality': None,
        'rejection_reason': None
    }

    if len(signal) < 10:
        result['rejection_reason'] = 'Signal too short'
        return result

    signal_std = np.std(signal)
    min_prominence = min_prominence_ratio * signal_std

    # Find positive peaks with minimum prominence
    pos_peaks, pos_props = find_peaks(signal,
                                      prominence=min_prominence,
                                      distance=int(fs * 0.2))
    if len(pos_peaks) == 0:
        result['rejection_reason'] = 'No positive peaks found'
        return result

    pos_prom = pos_props['prominences']

    # Find negative peaks with minimum prominence
    neg_peaks, neg_props = find_peaks(-signal,
                                      prominence=min_prominence,
                                      distance=int(fs * 0.2))
    if len(neg_peaks) == 0:
        result['rejection_reason'] = 'No negative peaks found'
        return result

    neg_prom = neg_props['prominences']

    # Get most prominent positive peak
    main_pos_idx = np.argmax(pos_prom)
    main_pos_peak = pos_peaks[main_pos_idx]
    main_pos_prom = pos_prom[main_pos_idx]

    # Find negative peak that comes after positive peak
    following_neg = neg_peaks[neg_peaks > main_pos_peak]
    if len(following_neg) == 0:
        result['rejection_reason'] = 'No negative peak after positive'
        return result

    # Time window: 0.08 - 0.7 seconds after positive peak
    time_diffs = following_neg - main_pos_peak
    valid_time_mask = (time_diffs >= int(0.08 * fs)) & (time_diffs <= int(0.7 * fs))
    following_neg_valid = following_neg[valid_time_mask]

    if len(following_neg_valid) == 0:
        result['rejection_reason'] = 'No negative peak in valid time window'
        return result

    # Get the negative peak with highest prominence in valid window
    valid_neg_indices = np.where(valid_time_mask)[0]
    valid_neg_prom = neg_prom[np.isin(neg_peaks, following_neg_valid)]
    main_neg_idx = valid_neg_indices[np.argmax(valid_neg_prom)]
    main_neg_peak = neg_peaks[main_neg_idx]
    main_neg_prom = neg_prom[main_neg_idx]

    # Check for multiple similar positive peaks (artifact indicator)
    similar_pos_peaks = np.sum(pos_prom >= main_pos_prom * 0.7)
    if similar_pos_peaks >= 2:
        result['rejection_reason'] = 'Multiple similar positive peaks (artifact)'
        return result

    # Check for multiple similar negative peaks in the range
    neg_in_range = neg_peaks[(neg_peaks > main_pos_peak) & (neg_peaks < main_neg_peak + int(0.3 * fs))]
    neg_prom_in_range = neg_prom[np.isin(neg_peaks, neg_in_range)]
    similar_neg_peaks = np.sum(neg_prom_in_range >= main_neg_prom * 0.7)
    if similar_neg_peaks >= 2:
        result['rejection_reason'] = 'Multiple similar negative peaks (artifact)'
        return result

    # Calculate characteristics
    pos_amplitude = signal[main_pos_peak]
    neg_amplitude = abs(signal[main_neg_peak])
    peak_to_peak = pos_amplitude + neg_amplitude
    duration = (main_neg_peak - main_pos_peak) / fs

    # Validate duration (already checked in window, but double-check)
    if duration < 0.08 or duration > 0.7:
        result['rejection_reason'] = f'Invalid duration: {duration:.3f}s'
        return result

    # Calculate shape quality
    shape_quality = calculate_shape_quality(signal, main_pos_peak, main_neg_peak, fs)

    # Require minimum shape quality
    if shape_quality['overall_quality'] < 0.5:
        result['rejection_reason'] = f'Poor shape quality: {shape_quality["overall_quality"]:.2f}'
        return result

    # Check peak balance (positive and negative should be similar)
    if shape_quality['peak_ratio'] < 0.3 or shape_quality['peak_ratio'] > 3.0:
        result['rejection_reason'] = f'Unbalanced peaks: ratio={shape_quality["peak_ratio"]:.2f}'
        return result

    # Artifact detection: check for unreasonable slope
    diff_signal = np.diff(signal)
    max_diff = np.max(np.abs(diff_signal))
    median_diff = np.median(np.abs(diff_signal))

    if max_diff > median_diff * 20:
        result['rejection_reason'] = 'Excessive slope (artifact)'
        return result

    # All checks passed
    result = {
        'has_valid_pattern': True,
        'pos_peak_idx': main_pos_peak,
        'neg_peak_idx': main_neg_peak,
        'pos_amplitude': pos_amplitude,
        'neg_amplitude': neg_amplitude,
        'peak_to_peak': peak_to_peak,
        'duration': duration,
        'shape_quality': shape_quality,
        'rejection_reason': None
    }

    return result


def validate_kcomplex_event_strict(raw_signal, event_start, event_end, fs=200,
                                   min_amplitude=75, max_amplitude=300,
                                   min_duration=0.3, max_duration=1.5,
                                   min_snr=2.5, min_shape_quality=0.6,
                                   check_context=True):
    """
    STRICT validation of K-complex event with clinical standards

    Args:
        raw_signal: 1D array (raw EEG)
        event_start: event start index
        event_end: event end index
        fs: sampling frequency
        min_amplitude: MINIMUM peak-to-peak amplitude (microvolts) - DEFAULT 75µV
        max_amplitude: maximum peak-to-peak amplitude (microvolts)
        min_duration: minimum total duration (seconds) - DEFAULT 0.3s
        max_duration: maximum total duration (seconds)
        min_snr: minimum signal-to-noise ratio - DEFAULT 2.5
        min_shape_quality: minimum overall shape quality score - DEFAULT 0.6
        check_context: whether to check surrounding baseline activity

    Returns:
        is_valid: bool
        info: dict with validation details
    """
    # Extract segment
    segment = raw_signal[event_start:event_end]

    if len(segment) < int(min_duration * fs):
        return False, {'reason': 'Too short', 'duration': len(segment) / fs}

    # Filter signal (0.5-4 Hz for K-complex)
    try:
        nyquist = fs / 2
        b, a = butter(4, [0.5 / nyquist, 4.0 / nyquist], btype='band')
        filtered_sig = filtfilt(b, a, segment)
    except:
        return False, {'reason': 'Filtering failed'}

    # 1. SNR check (STRICTER)
    noise_level = np.std(segment - filtered_sig)
    signal_level = np.std(filtered_sig)
    snr = signal_level / (noise_level + 1e-10)

    if snr < min_snr:
        return False, {'reason': 'Low SNR', 'snr': snr, 'min_required': min_snr}

    # 2. Context check: baseline activity (if enabled)
    if check_context:
        baseline_std, baseline_mean = calculate_baseline_noise(
            raw_signal, event_start, event_end, fs, window_sec=1.0
        )

        if not np.isnan(baseline_std):
            # Event should stand out from baseline
            event_amplitude = np.max(np.abs(filtered_sig))
            if event_amplitude < 2.0 * baseline_std:
                return False, {
                    'reason': 'Event does not stand out from baseline',
                    'event_amplitude': event_amplitude,
                    'baseline_std': baseline_std
                }

            # Baseline should be relatively quiet (not too much activity)
            if baseline_std > 30:  # µV
                return False, {
                    'reason': 'Baseline too noisy',
                    'baseline_std': baseline_std
                }

    # 3. STRICT peak detection with shape quality
    peak_info = detect_kcomplex_peaks_strict(filtered_sig, fs, min_prominence_ratio=0.2)

    if not peak_info['has_valid_pattern']:
        return False, {
            'reason': 'Invalid peak pattern',
            'details': peak_info['rejection_reason']
        }

    # 4. STRICT amplitude validation
    amplitude = peak_info['peak_to_peak']
    if amplitude < min_amplitude:
        return False, {
            'reason': 'Amplitude too low',
            'amplitude': amplitude,
            'min_required': min_amplitude
        }
    if amplitude > max_amplitude:
        return False, {
            'reason': 'Amplitude too high (artifact)',
            'amplitude': amplitude,
            'max_allowed': max_amplitude
        }

    # 5. Shape quality check
    shape_quality = peak_info['shape_quality']['overall_quality']
    if shape_quality < min_shape_quality:
        return False, {
            'reason': 'Poor shape quality',
            'quality': shape_quality,
            'min_required': min_shape_quality,
            'shape_details': peak_info['shape_quality']
        }

    # 6. Duration validation
    total_duration = (event_end - event_start) / fs
    if total_duration < min_duration:
        return False, {
            'reason': 'Duration too short',
            'duration': total_duration,
            'min_required': min_duration
        }
    if total_duration > max_duration:
        return False, {
            'reason': 'Duration too long',
            'duration': total_duration,
            'max_allowed': max_duration
        }

    # All checks passed - this is a HIGH QUALITY K-complex
    info = {
        'valid': True,
        'snr': snr,
        'amplitude': amplitude,
        'duration': total_duration,
        'peak_duration': peak_info['duration'],
        'shape_quality': shape_quality,
        'shape_details': peak_info['shape_quality'],
        'peak_info': peak_info
    }

    if check_context and not np.isnan(baseline_std):
        info['baseline_std'] = baseline_std
        info['baseline_mean'] = baseline_mean

    return True, info


def refine_event_boundaries_with_zerocrossing(signal, event_start, event_end, fs=200):
    """Refine event boundaries to align with zero-crossings"""
    zero_crossings = find_zero_crossings(signal, fs)

    if len(zero_crossings) == 0:
        return event_start, event_end

    # Refine start
    zc_before_start = zero_crossings[zero_crossings <= event_start]
    if len(zc_before_start) > 0:
        refined_start = zc_before_start[-1]
    else:
        refined_start = event_start

    # Refine end
    zc_after_end = zero_crossings[zero_crossings >= event_end]
    if len(zc_after_end) > 0:
        refined_end = zc_after_end[0]
    else:
        refined_end = event_end

    # Limit extension
    max_extension = int(0.3 * fs)
    if refined_start < event_start - max_extension:
        refined_start = event_start
    if refined_end > event_end + max_extension:
        refined_end = event_end

    return refined_start, refined_end


def postprocess_kcomplex_predictions_strict(predictions, raw_signal, fs=200,
                                            min_amplitude=75, max_amplitude=300,
                                            min_duration=0.3, max_duration=1.5,
                                            threshold=0.5, min_snr=2.5,
                                            min_shape_quality=0.6,
                                            check_context=True,
                                            refine_boundaries=True):
    """
    STRICT post-processing for K-complex with clinical standards

    Key differences from relaxed version:
    - min_amplitude: 75µV (was 15µV)
    - min_duration: 0.3s (was 0.15s)
    - min_snr: 2.5 (was 1.5)
    - min_shape_quality: 0.6 (new requirement)
    - check_context: True (new feature)

    Returns:
        refined_predictions: binary predictions
        events_info: list of validated events with quality metrics
    """
    # Threshold predictions
    binary_preds = (predictions >= threshold).astype(np.uint8)

    # Find events
    events = find_events(binary_preds)

    if len(events) == 0:
        return np.zeros_like(predictions, dtype=np.uint8), []

    # Filter signal for processing
    try:
        nyquist = fs / 2
        b, a = butter(4, [0.5 / nyquist, 4.0 / nyquist], btype='band')
        filtered_signal = filtfilt(b, a, raw_signal)
    except:
        filtered_signal = raw_signal

    # Process each event
    refined_predictions = np.zeros_like(predictions, dtype=np.uint8)
    events_info = []
    rejected_events = []

    for start, end in events:
        start = max(0, start)
        end = min(len(raw_signal) - 1, end)

        if end <= start:
            continue

        # Refine boundaries with zero-crossings
        if refine_boundaries:
            margin = int(0.5 * fs)
            search_start = max(0, start - margin)
            search_end = min(len(filtered_signal), end + margin)

            search_segment = filtered_signal[search_start:search_end]
            local_start = start - search_start
            local_end = end - search_start

            refined_local_start, refined_local_end = refine_event_boundaries_with_zerocrossing(
                search_segment, local_start, local_end, fs
            )

            refined_start = search_start + refined_local_start
            refined_end = search_start + refined_local_end
        else:
            refined_start = start
            refined_end = end

        # STRICT validation
        is_valid, info = validate_kcomplex_event_strict(
            raw_signal, refined_start, refined_end, fs,
            min_amplitude=min_amplitude,
            max_amplitude=max_amplitude,
            min_duration=min_duration,
            max_duration=max_duration,
            min_snr=min_snr,
            min_shape_quality=min_shape_quality,
            check_context=check_context
        )

        if is_valid:
            refined_predictions[refined_start:refined_end + 1] = 1
            events_info.append({
                'start': refined_start,
                'end': refined_end,
                'start_time': refined_start / fs,
                'end_time': refined_end / fs,
                **info
            })
        else:
            rejected_events.append({
                'start': refined_start,
                'end': refined_end,
                'start_time': refined_start / fs,
                'rejection_reason': info.get('reason', 'Unknown'),
                'rejection_details': info
            })

    print(f"K-complex detection summary:")
    print(f"  Candidate events: {len(events)}")
    print(f"  Validated K-complexes: {len(events_info)}")
    print(f"  Rejected: {len(rejected_events)}")

    if len(rejected_events) > 0:
        rejection_reasons = {}
        for evt in rejected_events:
            reason = evt['rejection_reason']
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        print("  Rejection breakdown:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {reason}: {count}")

    return refined_predictions, events_info


def find_events(binary_sequence):
    """Find event segments in binary sequence"""
    events = []
    in_event = False
    start = 0

    for i in range(len(binary_sequence)):
        if not in_event:
            if binary_sequence[i] == 1:
                in_event = True
                start = i
        else:
            if binary_sequence[i] == 0:
                end = i - 1
                events.append((start, end))
                in_event = False

    if in_event:
        events.append((start, len(binary_sequence) - 1))

    return events


# Example usage and testing
if __name__ == "__main__":
    print("STRICT K-Complex Detection - Clinical Standards")
    print("=" * 60)
    print("Parameters:")
    print("  Minimum amplitude: 75 µV (clinical standard)")
    print("  Minimum duration: 0.3 seconds")
    print("  Minimum SNR: 2.5")
    print("  Minimum shape quality: 0.6")
    print("  Context checking: ENABLED")
    print("=" * 60)

    # Generate synthetic data
    fs = 200
    duration = 10
    time = np.linspace(0, duration, int(duration * fs))

    # Background noise
    signal = np.random.randn(len(time)) * 5

    # Add a HIGH-QUALITY K-complex at t=5s
    kc_start = int(5 * fs)
    kc_duration = int(0.5 * fs)
    t_kc = np.linspace(0, 0.5, kc_duration)

    # Strong, clear biphasic waveform (100µV amplitude)
    kc_signal = 100 * np.sin(2 * np.pi * 2 * t_kc) * np.exp(-3 * t_kc)
    signal[kc_start:kc_start + kc_duration] += kc_signal

    # Predictions
    predictions = np.zeros(len(time))
    predictions[kc_start - 10:kc_start + kc_duration + 10] = 0.95

    # Post-process with STRICT criteria
    refined_preds, events_info = postprocess_kcomplex_predictions_strict(
        predictions, signal, fs=fs, threshold=0.5,
        min_amplitude=75, min_shape_quality=0.6, check_context=True
    )

    print(f"\nResults:")
    for i, event in enumerate(events_info):
        print(f"\nK-complex #{i + 1}:")
        print(f"  Time: {event['start_time']:.2f}s - {event['end_time']:.2f}s")
        print(f"  Duration: {event['duration']:.3f}s")
        print(f"  Amplitude: {event['amplitude']:.1f} µV")
        print(f"  SNR: {event['snr']:.2f}")
        print(f"  Shape quality: {event['shape_quality']:.2f}")
        if 'baseline_std' in event:
            print(f"  Baseline noise: {event['baseline_std']:.1f} µV")
