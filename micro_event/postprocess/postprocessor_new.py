import numpy as np
from scipy.signal import peak_prominences, hilbert, butter, filtfilt, find_peaks, savgol_filter
from scipy.stats import pearsonr
import numpy as np

def build_channel_index_map(dataset):
    """
    user_infos → {channel_name: [dataset_idx_0, dataset_idx_1, …]}
    한 번만 만들어 dataset 객체에 캐시한다.
    """
    mapping = {}
    for idx, (_sid, ch) in enumerate(dataset.user_infos):
        mapping.setdefault(ch, []).append(idx)
    return mapping


def get_raw_segment(
    dataset,
    channel: str,
    page_num: int,
    start25: int,
    stop25: int,
    use_raw: bool = True,
    *,
    raw_fs: int | None = None,
) -> np.ndarray:

    if raw_fs is None:
        raw_fs = dataset.fs            
    if raw_fs != dataset.fs:
        raise ValueError("현재 구현은 dataset.fs 와 동일한 raw_fs만 지원합니다.")

    if not hasattr(dataset, "_idx_map"):
        dataset._idx_map = build_channel_index_map(dataset)
    try:
        dataset_idx = dataset._idx_map[channel][page_num]
    except (KeyError, IndexError):
        raise IndexError(f"{channel} 의 page #{page_num} 가 dataset에 없습니다.")

    if use_raw:
        seg = dataset.raw_signals[dataset_idx]
    else:
        seg = dataset.signals[dataset_idx]    

    stride = dataset.stride                   
    bs = dataset.border_size                  

    raw_start = bs + start25 * stride
    raw_stop  = bs + stop25  * stride         

    raw_start = int(max(raw_start, 0))
    raw_stop  = int(min(raw_stop, seg.shape[-1]))
    if raw_start >= raw_stop:
        raise ValueError("start/stop 인덱스가 잘못되었습니다.")

    return seg[raw_start:raw_stop].copy()     

def find_events(sequence):
    events = []
    in_event = False
    start = 0
    length = len(sequence)
    for i in range(length):
        if not in_event:
            if sequence[i] == 1:
                in_event = True
                start = i
        else:
            if sequence[i] == 0:
                end = i - 1
                events.append((start, end))
                in_event = False

    if in_event:
        events.append((start, length - 1))

    return events


def validate_kcomplex(raw_seg, fs_raw, min_duration=0.15):
    # 1. 대역통과 필터 적용 (0.5-4Hz)
    nyquist = fs_raw / 2
    low = 0.5 / nyquist
    high = 4.0 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_sig = filtfilt(b, a, raw_seg)
    
    # 2. 신호 품질 체크 (SNR)
    noise_level = np.std(raw_seg - filtered_sig)
    signal_level = np.std(filtered_sig)
    snr = signal_level / (noise_level + 1e-10)
    
    if snr < 1.5:  # SNR이 너무 낮으면 제외
        return False, {'reason': 'Low SNR', 'snr': snr}
    
    # 3. Peak detection (negative & positive)
    # Negative peaks
    neg_peaks, _ = find_peaks(-filtered_sig, distance=int(fs_raw * 0.2))
    neg_prom = peak_prominences(-filtered_sig, neg_peaks)[0]
    
    # Positive peaks  
    pos_peaks, _ = find_peaks(filtered_sig, distance=int(fs_raw * 0.2))
    pos_prom = peak_prominences(filtered_sig, pos_peaks)[0]
    
    if len(neg_peaks) == 0 or len(pos_peaks) == 0:
        return False, {'reason': 'No peaks found'}
    
    # 4. 가장 prominent한 positive peak 찾기 (K-complex는 positive로 시작)
    main_pos_idx = np.argmax(pos_prom)
    main_pos_peak = pos_peaks[main_pos_idx]
    
    # 5. Positive peak 이후의 negative peak 찾기
    following_neg = neg_peaks[neg_peaks > main_pos_peak]
    if len(following_neg) == 0:
        return False, {'reason': 'No negative peak after positive'}
    
    # 5-1. Positive peak 이후 0.05초~0.5초 이내의 negative peak만 고려
    time_window_mask = (following_neg - main_pos_peak) <= int(1.0 * fs_raw)
    following_neg_in_window = following_neg[time_window_mask]
    
    if len(following_neg_in_window) == 0:
        return False, {'reason': 'No negative peak in valid time window'}
    
    # 가장 가까운 negative peak (시간 창 내에서)
    main_neg_peak = following_neg_in_window[0]
    
    # 6. Multiple positive peaks 체크
    # 전체 positive peak들의 amplitude 확인
    pos_amplitudes = filtered_sig[pos_peaks]
    max_pos_amplitude = np.max(pos_amplitudes)
    
    # 최대 amplitude의 70% 이상인 positive peak 개수 확인
    similar_peaks = np.sum(pos_amplitudes >= max_pos_amplitude * 0.7)
    
    if similar_peaks >= 2:
        return False, {'reason': 'Multiple similar positive peaks', 'similar_peaks': similar_peaks, 
                       'all_pos_peaks': pos_peaks, 'all_pos_amplitudes': pos_amplitudes, 'neg_peak_pos': main_neg_peak, 'pos_peak_pos': main_pos_peak}
    
    # 7. 지속시간 체크 (positive peak에서 negative peak까지)
    duration = (main_neg_peak - main_pos_peak) / fs_raw
    if duration < 0.08 or duration > 0.7:
        return False, {'reason': 'Invalid duration', 'duration': duration,
                       'all_pos_peaks': pos_peaks, 'all_pos_amplitudes': pos_amplitudes, 'neg_peak_pos': main_neg_peak, 'pos_peak_pos': main_pos_peak}
    
    # 8. 진폭 계산 (제한은 없음)
    pos_amplitude = filtered_sig[main_pos_peak]
    neg_amplitude = abs(filtered_sig[main_neg_peak])
    peak_to_peak = pos_amplitude + neg_amplitude
    
    # 10. 전체 K-complex 지속시간 체크
    # K-complex의 시작과 끝 찾기 (positive peak부터 시작)
    kc_start = max(0, main_pos_peak - int(fs_raw * 0.1))
    kc_end = min(len(filtered_sig) - 1, main_neg_peak + int(fs_raw * 0.2))
    total_duration = (kc_end - kc_start) / fs_raw
    
    if total_duration < min_duration:
        return False, {'reason': 'Too short total duration', 'total_duration': total_duration}
    
    # 11. 급격한 변화 체크 (아티팩트 제거)
    diff_signal = np.diff(filtered_sig)
    max_diff = np.max(np.abs(diff_signal))
    median_diff = np.median(np.abs(diff_signal))
    
    if max_diff > median_diff * 20 or peak_to_peak > 250:  # 너무 급격한 변화는 아티팩트
        return False, {'reason': 'Artifact detected'}
    
    features = {
        'valid': True,
        'pos_peak_pos': main_pos_peak,  # positive가 먼저
        'neg_peak_pos': main_neg_peak,  # negative가 나중
        'duration': duration,
        'amplitude': peak_to_peak,
        'pos_amplitude': pos_amplitude,
        'neg_amplitude': neg_amplitude,
        'snr': snr,
        'total_duration': total_duration,
        'num_pos_peaks': len(pos_peaks),
        'max_pos_amplitude': max_pos_amplitude,
        'all_pos_peaks': pos_peaks,
        'all_pos_amplitudes': pos_amplitudes
    }
    
    return True, features

def detect_spindle_characteristics(raw_seg, fs_raw):
    b, a = butter(4, [11/(fs_raw/2), 16/(fs_raw/2)], btype='band')
    filt = filtfilt(b, a, raw_seg)
    
    analytic_signal = hilbert(filt)
    envelope = np.abs(analytic_signal)
    instantaneous_phase = np.angle(analytic_signal)
    instantaneous_freq = np.diff(np.unwrap(instantaneous_phase)) / (2.0*np.pi) * fs_raw
    
    if len(envelope) > 51:
        smooth_envelope = savgol_filter(envelope, 51, 3)
    else:
        smooth_envelope = envelope
    
    valid_freq_mask = (instantaneous_freq > 10) & (instantaneous_freq < 17)
    if np.sum(valid_freq_mask) > 0:
        mean_freq = np.mean(instantaneous_freq[valid_freq_mask])
        freq_std = np.std(instantaneous_freq[valid_freq_mask])
    else:
        mean_freq = 0
        freq_std = 999
    
    is_spindle_freq = (12 <= mean_freq <= 15) and (freq_std < 2.0)
    
    envelope_normalized = (smooth_envelope - smooth_envelope.min()) / (smooth_envelope.max() - smooth_envelope.min() + 1e-8)
    
    max_idx = np.argmax(envelope_normalized)
    total_len = len(envelope_normalized)
    
    is_centered = 0.2 <= (max_idx / total_len) <= 0.8
    
    if is_centered and max_idx > 10 and max_idx < total_len - 10:
        first_part = envelope_normalized[:max_idx]
        second_part = envelope_normalized[max_idx:]
        
        if len(first_part) > 5 and len(second_part) > 5:
            first_trend = pearsonr(np.arange(len(first_part)), first_part)[0]
            second_trend = pearsonr(np.arange(len(second_part)), second_part)[0]
            waxing_waning = first_trend > 0.5 and second_trend < -0.5
        else:
            waxing_waning = False
    else:
        waxing_waning = False
    
    zero_crossings = np.where(np.diff(np.signbit(filt)))[0]
    if len(zero_crossings) > 4:
        zc_intervals = np.diff(zero_crossings)
        zc_regularity = np.std(zc_intervals) / (np.mean(zc_intervals) + 1e-8)
        is_regular = zc_regularity < 0.3
        
        zc_freq = fs_raw / (2 * np.mean(zc_intervals))
        is_zc_spindle_freq = 12 <= zc_freq <= 15
    else:
        is_regular = False
        is_zc_spindle_freq = False
        zc_regularity = 999
    
    peaks, properties = find_peaks(smooth_envelope, 
                                   prominence=np.std(smooth_envelope) * 0.5,
                                   distance=int(fs_raw * 0.05))
    
    if len(peaks) >= 5:
        peak_amplitudes = smooth_envelope[peaks]

        amplitude_cv = np.std(peak_amplitudes) / (np.mean(peak_amplitudes) + 1e-8)
        is_amplitude_consistent = amplitude_cv < 0.4
    else:
        is_amplitude_consistent = False
        amplitude_cv = 999
    
    b_bg, a_bg = butter(4, [5/(fs_raw/2), 10/(fs_raw/2)], btype='band')
    background = filtfilt(b_bg, a_bg, raw_seg)
    
    signal_power = np.mean(filt**2)
    background_power = np.mean(background**2)
    snr = 10 * np.log10(signal_power / (background_power + 1e-8))
    
    is_high_snr = snr > 3
    
    duration = len(raw_seg) / fs_raw
    is_valid_duration = 0.5 <= duration <= 2.8
    
    fft_vals = np.fft.fft(filt)
    fft_freq = np.fft.fftfreq(len(filt), 1/fs_raw)
    
    spindle_band_mask = (np.abs(fft_freq) >= 11) & (np.abs(fft_freq) <= 16)
    spindle_power = np.sum(np.abs(fft_vals[spindle_band_mask])**2)
    
    total_mask = (np.abs(fft_freq) >= 0) & (np.abs(fft_freq) <= 30)
    total_power = np.sum(np.abs(fft_vals[total_mask])**2)
    
    spectral_concentration = spindle_power / (total_power + 1e-8)
    is_concentrated = spectral_concentration > 0.5
    
    criteria_scores = {
        'freq_range': is_spindle_freq * 2,  # 중요
        'waxing_waning': waxing_waning * 2,  # 중요
        'regularity': is_regular * 1.5,
        'zc_freq': is_zc_spindle_freq * 1,
        'amplitude_consistency': is_amplitude_consistent * 1,
        'snr': is_high_snr * 1.5,
        'duration': is_valid_duration * 1,
        'spectral_concentration': is_concentrated * 1.5
    }
    
    total_score = sum(criteria_scores.values())
    max_score = sum([2, 2, 1.5, 1, 1, 1.5, 1, 1.5])  # 11.5
    
    is_spindle = total_score >= (0.7 * max_score)
    
    essential_criteria = is_spindle_freq and waxing_waning and is_valid_duration
    # is_spindle = is_spindle and essential_criteria
    
    spindle_info = {
        'is_spindle': is_spindle,
        'duration': duration,
        'mean_freq': mean_freq,
        'freq_std': freq_std,
        'waxing_waning': waxing_waning,
        'is_valid_duration': is_valid_duration,
        'is_spindle_freq': is_spindle_freq,
        'regularity': is_regular,
        'zc_regularity': zc_regularity,
        'amplitude_consistency': is_amplitude_consistent,
        'amplitude_cv': amplitude_cv,
        'snr': snr,
        'spectral_concentration': spectral_concentration,
        'confidence_score': total_score,
        'confidence': total_score,
        'max_score': max_score,
        'envelope': smooth_envelope,
        'filtered_signal': filt,
        'peaks': peaks,
        'criteria_scores': criteria_scores
    }
    
    return spindle_info

def analyze_kcomplex_events(events, channel, event_type, fs_pred, fs_raw, 
                           sleep_dataset, page_duration):
    validated_events = []
    invalid_events = []
    all_results = []
    
    print(f"\n{channel} - Validating {event_type}s...")
    
    for st, ed in events:
        # pred 인덱스 → raw 신호 인덱스 변환
        t0 = st / fs_pred
        page_idx = int(t0 // page_duration)
        offset_s = t0 % page_duration
        raw_start = int(offset_s * fs_raw)
        raw_end = raw_start + int((ed - st) / fs_pred * fs_raw)
        
        # 전후 여백 추가
        margin = int(0.5 * fs_raw)
        raw_start_ext = max(0, raw_start - margin)
        raw_end_ext = raw_end + margin
        
        raw_seg = get_raw_segment(
            sleep_dataset, channel, page_idx, raw_start_ext//8, raw_end_ext//8
        )
        
        # K-complex 검증
        is_valid, features = validate_kcomplex(raw_seg, fs_raw)
        all_results.append((st, ed, is_valid, features))
        
        if is_valid:
            validated_events.append((st, ed))
        else:
            invalid_events.append((st, ed, features))

    return validated_events, all_results, invalid_events


def postprocess_preds(preds_all, sleep_dataset, 
                      event_type, page_duration,
                      fs_pred=200//8, fs_raw=200):
    refined_preds = {}
    for channel, preds in preds_all.items():
        refined_pred = np.zeros_like(preds, dtype=np.uint8)
        events = find_events(preds)  # [(st, ed), ...]

        if event_type == 'kcomplex':
            validated_events, results, invalid_events = analyze_kcomplex_events(
                events, channel, 'False Positive', 
                fs_pred, fs_raw, sleep_dataset, page_duration
            )

            for st, ed in validated_events:
                refined_pred[st:ed] = 1
        else:
            for st, ed in events:
                t0 = st / fs_pred
                page_idx = int(t0 // page_duration)
                offset_s = t0 % page_duration
                rs = int(offset_s * fs_raw)
                re = rs + int((ed - st) / fs_pred * fs_raw)
                raw_seg = get_raw_segment(sleep_dataset, channel, page_idx, rs//8, re//8)

                spindle_info = detect_spindle_characteristics(raw_seg, fs_raw)
                if spindle_info['is_spindle']:
                    refined_pred[st:ed] = 1

        refined_preds[channel] = refined_pred

    return refined_preds