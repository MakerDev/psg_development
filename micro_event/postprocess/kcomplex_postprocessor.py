"""
K-Complex Post-processing with Zero-Crossing Detection

This module provides advanced post-processing for K-complex detection:
1. Zero-crossing boundary refinement
2. Peak detection and validation
3. Amplitude filtering
4. Duration filtering
"""

import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences


def find_zero_crossings(signal, fs=200):
    """
    Find zero-crossing points in a signal

    Args:
        signal: 1D array
        fs: sampling frequency

    Returns:
        indices of zero-crossings
    """
    # Sign changes indicate zero crossings
    sign = np.sign(signal)
    sign[sign == 0] = -1  # treat zeros as negative
    zero_crossings = np.where(np.diff(sign))[0]
    return zero_crossings


def refine_event_boundaries_with_zerocrossing(signal, event_start, event_end, fs=200):
    """
    Refine event boundaries to align with zero-crossings

    K-complex boundaries should be at zero-crossing points:
    - Start: zero-crossing before the positive peak
    - End: zero-crossing after the negative peak

    Args:
        signal: 1D array (filtered signal)
        event_start: initial event start index
        event_end: initial event end index
        fs: sampling frequency

    Returns:
        refined_start, refined_end
    """
    # Find zero crossings
    zero_crossings = find_zero_crossings(signal, fs)

    if len(zero_crossings) == 0:
        return event_start, event_end

    # Refine start: find nearest zero-crossing before event_start
    zc_before_start = zero_crossings[zero_crossings <= event_start]
    if len(zc_before_start) > 0:
        refined_start = zc_before_start[-1]  # closest to event_start
    else:
        refined_start = event_start

    # Refine end: find nearest zero-crossing after event_end
    zc_after_end = zero_crossings[zero_crossings >= event_end]
    if len(zc_after_end) > 0:
        refined_end = zc_after_end[0]  # closest to event_end
    else:
        refined_end = event_end

    # Ensure refined boundaries are reasonable
    max_extension = int(0.3 * fs)  # don't extend more than 300ms
    if refined_start < event_start - max_extension:
        refined_start = event_start
    if refined_end > event_end + max_extension:
        refined_end = event_end

    return refined_start, refined_end


def detect_kcomplex_peaks(signal, fs=200):
    """
    Detect positive and negative peaks in a signal segment

    Returns characteristics of the most prominent positive and negative peaks.

    Args:
        signal: 1D array (should be filtered 0.5-4 Hz)
        fs: sampling frequency

    Returns:
        dict with keys:
            - has_valid_pattern: bool
            - pos_peak_idx: int
            - neg_peak_idx: int
            - pos_amplitude: float
            - neg_amplitude: float
            - peak_to_peak: float
            - duration: float (seconds)
    """
    result = {
        'has_valid_pattern': False,
        'pos_peak_idx': None,
        'neg_peak_idx': None,
        'pos_amplitude': 0,
        'neg_amplitude': 0,
        'peak_to_peak': 0,
        'duration': 0
    }

    if len(signal) < 10:
        return result

    # Find positive peaks
    pos_peaks, _ = find_peaks(signal, distance=int(fs * 0.2))
    if len(pos_peaks) == 0:
        return result

    pos_prom = peak_prominences(signal, pos_peaks)[0]

    # Find negative peaks
    neg_peaks, _ = find_peaks(-signal, distance=int(fs * 0.2))
    if len(neg_peaks) == 0:
        return result

    neg_prom = peak_prominences(-signal, neg_peaks)[0]

    # Get most prominent peaks
    main_pos_idx = np.argmax(pos_prom)
    main_pos_peak = pos_peaks[main_pos_idx]

    # Find negative peak that comes after positive peak
    following_neg = neg_peaks[neg_peaks > main_pos_peak]
    if len(following_neg) == 0:
        return result

    # Check time window (0.05 - 1.0 seconds after positive peak)
    time_window_mask = (following_neg - main_pos_peak) <= int(1.0 * fs)
    following_neg_in_window = following_neg[time_window_mask]

    if len(following_neg_in_window) == 0:
        return result

    # Get closest negative peak
    main_neg_peak = following_neg_in_window[0]

    # Check for multiple similar positive peaks (artifact indicator)
    pos_amplitudes = signal[pos_peaks]
    max_pos_amplitude = np.max(pos_amplitudes)
    similar_peaks = np.sum(pos_amplitudes >= max_pos_amplitude * 0.7)

    if similar_peaks >= 2:
        return result

    # Calculate characteristics
    pos_amplitude = signal[main_pos_peak]
    neg_amplitude = abs(signal[main_neg_peak])
    peak_to_peak = pos_amplitude + neg_amplitude
    duration = (main_neg_peak - main_pos_peak) / fs

    # Validate duration
    if duration < 0.08 or duration > 0.7:
        return result

    # Validate amplitude (avoid artifacts)
    diff_signal = np.diff(signal)
    max_diff = np.max(np.abs(diff_signal))
    median_diff = np.median(np.abs(diff_signal))

    if max_diff > median_diff * 20 or peak_to_peak > 250:
        return result

    result = {
        'has_valid_pattern': True,
        'pos_peak_idx': main_pos_peak,
        'neg_peak_idx': main_neg_peak,
        'pos_amplitude': pos_amplitude,
        'neg_amplitude': neg_amplitude,
        'peak_to_peak': peak_to_peak,
        'duration': duration
    }

    return result


def validate_kcomplex_event(raw_signal, event_start, event_end, fs=200,
                            min_amplitude=15, max_amplitude=250,
                            min_duration=0.15, max_duration=1.5):
    """
    Comprehensive validation of a K-complex event

    Args:
        raw_signal: 1D array (raw EEG)
        event_start: event start index
        event_end: event end index
        fs: sampling frequency
        min_amplitude: minimum peak-to-peak amplitude (microvolts)
        max_amplitude: maximum peak-to-peak amplitude (microvolts)
        min_duration: minimum total duration (seconds)
        max_duration: maximum total duration (seconds)

    Returns:
        is_valid: bool
        info: dict with validation details
    """
    # Extract segment
    segment = raw_signal[event_start:event_end]

    if len(segment) < int(min_duration * fs):
        return False, {'reason': 'Too short'}

    # Filter signal (0.5-4 Hz for K-complex)
    try:
        nyquist = fs / 2
        b, a = butter(4, [0.5 / nyquist, 4.0 / nyquist], btype='band')
        filtered_sig = filtfilt(b, a, segment)
    except:
        return False, {'reason': 'Filtering failed'}

    # 1. SNR check
    noise_level = np.std(segment - filtered_sig)
    signal_level = np.std(filtered_sig)
    snr = signal_level / (noise_level + 1e-10)

    if snr < 1.5:
        return False, {'reason': 'Low SNR', 'snr': snr}

    # 2. Peak detection
    peak_info = detect_kcomplex_peaks(filtered_sig, fs)

    if not peak_info['has_valid_pattern']:
        return False, {'reason': 'Invalid peak pattern'}

    # 3. Amplitude validation
    amplitude = peak_info['peak_to_peak']
    if amplitude < min_amplitude:
        return False, {'reason': 'Amplitude too low', 'amplitude': amplitude}
    if amplitude > max_amplitude:
        return False, {'reason': 'Amplitude too high', 'amplitude': amplitude}

    # 4. Duration validation
    total_duration = (event_end - event_start) / fs
    if total_duration < min_duration:
        return False, {'reason': 'Duration too short', 'duration': total_duration}
    if total_duration > max_duration:
        return False, {'reason': 'Duration too long', 'duration': total_duration}

    # All checks passed
    info = {
        'valid': True,
        'snr': snr,
        'amplitude': amplitude,
        'duration': total_duration,
        'peak_info': peak_info
    }

    return True, info


def postprocess_kcomplex_predictions(predictions, raw_signal, fs=200,
                                     min_amplitude=15, max_amplitude=250,
                                     min_duration=0.15, max_duration=1.5,
                                     threshold=0.5,
                                     refine_boundaries=True):
    """
    Post-process K-complex predictions with zero-crossing refinement

    Args:
        predictions: (time,) - model output probabilities
        raw_signal: (time,) - raw EEG signal
        fs: sampling frequency
        min_amplitude: minimum peak-to-peak amplitude
        max_amplitude: maximum peak-to-peak amplitude
        min_duration: minimum total duration
        max_duration: maximum total duration
        threshold: probability threshold for detection
        refine_boundaries: whether to refine boundaries with zero-crossings

    Returns:
        refined_predictions: (time,) - binary predictions after post-processing
        events_info: list of dicts with event information
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

    for start, end in events:
        # Ensure indices are within bounds
        start = max(0, start)
        end = min(len(raw_signal) - 1, end)

        if end <= start:
            continue

        # Refine boundaries with zero-crossings
        if refine_boundaries:
            # Add margin for boundary search
            margin = int(0.5 * fs)
            search_start = max(0, start - margin)
            search_end = min(len(filtered_signal), end + margin)

            search_segment = filtered_signal[search_start:search_end]
            local_start = start - search_start
            local_end = end - search_start

            refined_local_start, refined_local_end = refine_event_boundaries_with_zerocrossing(
                search_segment, local_start, local_end, fs
            )

            # Map back to global indices
            refined_start = search_start + refined_local_start
            refined_end = search_start + refined_local_end
        else:
            refined_start = start
            refined_end = end

        # Validate event
        is_valid, info = validate_kcomplex_event(
            raw_signal, refined_start, refined_end, fs,
            min_amplitude, max_amplitude, min_duration, max_duration
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

    return refined_predictions, events_info


def find_events(binary_sequence):
    """
    Find event segments in binary sequence

    Args:
        binary_sequence: 1D array of 0s and 1s

    Returns:
        list of (start, end) tuples
    """
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


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    fs = 200
    duration = 10  # seconds
    time = np.linspace(0, duration, int(duration * fs))

    # Synthetic K-complex signal
    signal = np.random.randn(len(time)) * 5  # background noise

    # Add a synthetic K-complex at t=5s
    kc_start = int(5 * fs)
    kc_duration = int(0.5 * fs)  # 500ms
    t_kc = np.linspace(0, 0.5, kc_duration)

    # Positive peak followed by negative peak
    kc_signal = 50 * np.sin(2 * np.pi * 2 * t_kc) * np.exp(-3 * t_kc)
    signal[kc_start:kc_start + kc_duration] += kc_signal

    # Synthetic predictions (high probability around K-complex)
    predictions = np.zeros(len(time))
    predictions[kc_start - 10:kc_start + kc_duration + 10] = 0.9

    # Post-process
    refined_preds, events_info = postprocess_kcomplex_predictions(
        predictions, signal, fs=fs, threshold=0.5, refine_boundaries=True
    )

    print(f"Original events: {np.sum(predictions > 0.5)} samples")
    print(f"Refined events: {np.sum(refined_preds)} samples")
    print(f"Number of validated K-complexes: {len(events_info)}")

    for i, event in enumerate(events_info):
        print(f"\nEvent {i + 1}:")
        print(f"  Time: {event['start_time']:.2f}s - {event['end_time']:.2f}s")
        print(f"  Duration: {event['duration']:.3f}s")
        print(f"  Amplitude: {event['amplitude']:.1f} µV")
        print(f"  SNR: {event['snr']:.2f}")
