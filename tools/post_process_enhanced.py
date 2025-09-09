import numpy as np
from typing import List, Tuple, Dict, NamedTuple

class PostProcessInfo(NamedTuple):
    epoch_idx: int
    original_stage: int
    corrected_stage: int
    was_changed: bool
    change_reason: str

def correct_sleep_stages_with_tracking(arousal_preds: List[Tuple[int, int]], 
                                     sleep_preds: List[int], 
                                     micro_event_preds_by_channels: Dict[str, np.ndarray]) -> Tuple[List[int], List[PostProcessInfo]]:
    """
    Correct sleep stages based on arousal and K-complex (micro event) information with detailed tracking.
    
    주요 보정 룰:
    1. K-complex가 arousal과 연관되지 않고 epoch 전반부에 있으면 N2
    2. Arousal과 연관된 K-complex는 N2의 증거가 아님
    3. Arousal 직후 epoch는 N1
    4. K-complex가 후반부에만 있으면 N2로 변경하지 않음
    
    Args:
        arousal_preds: List of (start_idx, end_idx) tuples for arousal events (50Hz)
        sleep_preds: List of sleep stages (0-4), one per 30-second epoch (50Hz)
        micro_event_preds_by_channels: Dict of channel_name -> binary array (25Hz)
    
    Returns:
        Tuple of (corrected sleep_preds list, post-processing info list)
    """
    
    # Constants
    EPOCH_DURATION_SEC = 30
    SAMPLES_PER_EPOCH_50HZ = 50 * EPOCH_DURATION_SEC  # 1500 samples
    HALF_EPOCH_50HZ = SAMPLES_PER_EPOCH_50HZ // 2  # 750 samples (15 seconds)
    AROUSAL_ASSOCIATION_WINDOW = 25  # 0.5 seconds at 50Hz
    
    # Sleep stage constants
    N1 = 2
    N2 = 3

    total_length = len(sleep_preds) * SAMPLES_PER_EPOCH_50HZ
    
    # Step 1: Integrate micro events across all channels (25Hz)
    integrated_micro_events_25hz = integrate_micro_events(micro_event_preds_by_channels)
    
    # Step 2: Convert micro events from 25Hz to 50Hz
    integrated_micro_events_50hz = upsample_25hz_to_50hz(integrated_micro_events_25hz)
    
    # Step 3: Detect K-complexes (continuous micro events)
    k_complexes = detect_k_complexes(integrated_micro_events_50hz)
    
    # Step 4: Create arousal mask for easier lookup
    arousal_mask = create_arousal_mask(arousal_preds, total_length)
    
    # Step 5: Correct sleep stages with tracking
    corrected_sleep_preds = sleep_preds.copy()
    post_process_info = []
    num_epochs = len(sleep_preds)
    
    for epoch_idx in range(num_epochs):
        epoch_start = epoch_idx * SAMPLES_PER_EPOCH_50HZ
        epoch_end = (epoch_idx + 1) * SAMPLES_PER_EPOCH_50HZ
        epoch_mid = epoch_start + HALF_EPOCH_50HZ
        
        # Get current stage
        original_stage = sleep_preds[epoch_idx]
        current_stage = corrected_sleep_preds[epoch_idx]
        change_reason = "NO_CHANGE"

        if current_stage not in [N1, N2]:
            post_process_info.append(PostProcessInfo(
                epoch_idx=epoch_idx,
                original_stage=original_stage,
                corrected_stage=current_stage,
                was_changed=False,
                change_reason="NOT_N1_N2"
            ))
            continue
        
        # Find K-complexes in this epoch
        k_complexes_in_epoch = []
        for k_start, k_end in k_complexes:
            # K-complex가 이 epoch와 겹치는지 확인
            if k_start < epoch_end and k_end > epoch_start:
                # Epoch 내에서의 K-complex 위치 계산
                k_start_in_epoch = max(k_start, epoch_start) - epoch_start
                k_end_in_epoch = min(k_end, epoch_end) - epoch_start
                k_complexes_in_epoch.append((k_start_in_epoch, k_end_in_epoch, k_start, k_end))
        
        # Rule 2: K-complex 기반 N2 scoring
        if k_complexes_in_epoch:
            # 전반부에 K-complex가 있는지 확인
            k_complex_in_first_half = False
            k_complex_associated_with_arousal = False
            
            for k_start_in_epoch, k_end_in_epoch, k_start_global, k_end_global in k_complexes_in_epoch:
                # K-complex가 전반부에 있는지 확인
                if k_start_in_epoch < HALF_EPOCH_50HZ:
                    k_complex_in_first_half = True
                    
                    # 이 K-complex가 arousal과 연관되어 있는지 확인 (0.5초 이내)
                    # K-complex 전후 0.5초 window 확인
                    window_start = max(0, k_start_global - AROUSAL_ASSOCIATION_WINDOW)
                    window_end = min(len(arousal_mask), k_end_global + AROUSAL_ASSOCIATION_WINDOW)
                    
                    if np.any(arousal_mask[window_start:window_end]):
                        k_complex_associated_with_arousal = True
            
            # K-complex가 전반부에 있고 arousal과 연관되지 않았다면 N2
            if k_complex_in_first_half and not k_complex_associated_with_arousal:
                if current_stage != N2:
                    corrected_sleep_preds[epoch_idx] = N2
                    change_reason = "KCOMPLEX_TO_N2"
        
        # Rule 3: 현재 epoch에 arousal이 있는 경우 처리
        if np.any(arousal_mask[epoch_start:epoch_end]):
            # Arousal 위치 찾기
            arousal_indices = np.where(arousal_mask[epoch_start:epoch_end])[0]
            first_arousal_in_epoch = arousal_indices[0]
            
            # Arousal이 후반부에 있고, 전반부에 유효한 K-complex가 있으면 N2 유지 가능
            if first_arousal_in_epoch >= HALF_EPOCH_50HZ:
                # 전반부에 arousal과 연관되지 않은 K-complex가 있는지 확인
                valid_k_in_first_half = False
                for k_start_in_epoch, k_end_in_epoch, k_start_global, k_end_global in k_complexes_in_epoch:
                    if k_end_in_epoch <= HALF_EPOCH_50HZ:  # 전반부에 완전히 포함
                        # Arousal과 연관되지 않았는지 확인
                        window_start = max(0, k_start_global - AROUSAL_ASSOCIATION_WINDOW)
                        window_end = min(len(arousal_mask), k_end_global + AROUSAL_ASSOCIATION_WINDOW)
                        if not np.any(arousal_mask[window_start:window_end]):
                            valid_k_in_first_half = True
                            break
                
                if valid_k_in_first_half:
                    if current_stage != N2:
                        corrected_sleep_preds[epoch_idx] = N2
                        change_reason = "AROUSAL_LATE_KCOMPLEX_TO_N2"
        
        # Record post-processing info
        final_stage = corrected_sleep_preds[epoch_idx]
        was_changed = (original_stage != final_stage)
        
        if was_changed and change_reason == "NO_CHANGE":
            # Determine actual change reason
            if final_stage == N2:
                change_reason = "KCOMPLEX_TO_N2"
            elif final_stage == N1:
                change_reason = "AROUSAL_TO_N1"
        
        post_process_info.append(PostProcessInfo(
            epoch_idx=epoch_idx,
            original_stage=original_stage,
            corrected_stage=final_stage,
            was_changed=was_changed,
            change_reason=change_reason
        ))
    
    return corrected_sleep_preds, post_process_info


def integrate_micro_events(micro_event_preds_by_channels: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Integrate micro events across all channels using logical OR.
    If any channel has an event at a time point, the integrated array has an event.
    
    Args:
        micro_event_preds_by_channels: Dict of channel_name -> binary array (25Hz)
    
    Returns:
        Integrated binary array (25Hz)
    """
    if not micro_event_preds_by_channels:
        return np.array([])
    
    # Get the length from any channel
    length = len(next(iter(micro_event_preds_by_channels.values())))
    integrated = np.zeros(length, dtype=bool)
    
    # OR operation across all channels
    for channel_data in micro_event_preds_by_channels.values():
        integrated |= channel_data.astype(bool)
    
    return integrated.astype(int)


def upsample_25hz_to_50hz(data_25hz: np.ndarray) -> np.ndarray:
    """
    Upsample data from 25Hz to 50Hz by repeating each sample twice.
    
    Args:
        data_25hz: Binary array at 25Hz
    
    Returns:
        Binary array at 50Hz
    """
    # Each 25Hz sample becomes two 50Hz samples
    data_50hz = np.repeat(data_25hz, 2)
    return data_50hz


def detect_k_complexes(micro_events_50hz: np.ndarray, min_duration_samples: int = 25) -> List[Tuple[int, int]]:
    """
    Detect K-complexes as continuous micro events.
    A K-complex is defined as a continuous period of micro events.
    
    Args:
        micro_events_50hz: Binary array of micro events (50Hz)
        min_duration_samples: Minimum duration for K-complex (default 25 = 0.5 seconds)
    
    Returns:
        List of (start_idx, end_idx) tuples for K-complexes
    """
    k_complexes = []
    
    if len(micro_events_50hz) == 0:
        return k_complexes
    
    # Find transitions
    diff = np.diff(np.concatenate([[0], micro_events_50hz, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Filter by minimum duration
    for start, end in zip(starts, ends):
        if end - start >= min_duration_samples:
            k_complexes.append((start, end))
    
    return k_complexes


def create_arousal_mask(arousal_preds: List[Tuple[int, int]], length: int) -> np.ndarray:
    """
    Create a binary mask for arousal events.
    
    Args:
        arousal_preds: List of (start_idx, end_idx) tuples for arousal events
        length: Total length of the mask
    
    Returns:
        Binary mask where 1 indicates arousal
    """
    mask = np.zeros(length, dtype=bool)
    
    for start, end in arousal_preds:
        start = int(start * 50)
        end = int(start + end * 50)

        if end >= length:
            break
        if start < length:
            mask[start:min(end, length)] = True
    
    return mask