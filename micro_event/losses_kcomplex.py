"""
K-Complex Specific Loss Functions

This module provides loss functions that incorporate K-complex morphological characteristics:
1. Shape-aware loss: Encourages correct positive→negative peak pattern
2. Peak alignment loss: Ensures peaks are detected within events
3. Duration constraint loss: Enforces valid K-complex duration
4. Zero-crossing boundary loss: Aligns event boundaries with zero-crossings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import butter, filtfilt


def masked_focal_loss(logits, targets, mask, alpha=0.25, gamma=2.0):
    """
    Standard masked focal loss for binary classification

    Args:
        logits: (batch, time, 2) - model outputs
        targets: (batch, time) - ground truth (0 or 1)
        mask: (batch, time) - valid positions (1=valid, 0=ignore)
        alpha: balancing factor for positive class
        gamma: focusing parameter
    """
    B, T, C = logits.shape
    logits = logits.reshape(-1, C)
    targets = targets.reshape(-1).long()
    mask = mask.reshape(-1)

    # One-hot encoding
    targets_onehot = F.one_hot(targets, num_classes=C).float()

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # Focal factor
    pt = (probs * targets_onehot).sum(dim=-1)
    alpha_t = alpha * targets_onehot[:, 1] + (1 - alpha) * targets_onehot[:, 0]
    focal_factor = alpha_t * (1 - pt) ** gamma

    ce = -(targets_onehot * log_probs).sum(dim=-1)
    loss = focal_factor * ce

    # Apply mask
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)


def peak_alignment_loss(peak_probs, targets, mask):
    """
    Encourage peak detection within K-complex events

    Args:
        peak_probs: (batch, time, 2) - [pos_peak_prob, neg_peak_prob]
        targets: (batch, time) - K-complex labels (0 or 1)
        mask: (batch, time) - valid positions

    Logic:
    - Within K-complex events (target=1): maximize peak probabilities
    - Outside events (target=0): minimize peak probabilities
    """
    B, T = targets.shape

    # Extract positive and negative peak probabilities
    pos_peak = peak_probs[:, :, 0]  # (batch, time)
    neg_peak = peak_probs[:, :, 1]  # (batch, time)

    # Loss for positive peaks
    # Within events: should have high probability
    # Outside events: should have low probability
    pos_loss_in = -torch.log(pos_peak + 1e-8) * targets * mask
    pos_loss_out = -torch.log(1 - pos_peak + 1e-8) * (1 - targets) * mask

    # Loss for negative peaks
    neg_loss_in = -torch.log(neg_peak + 1e-8) * targets * mask
    neg_loss_out = -torch.log(1 - neg_peak + 1e-8) * (1 - targets) * mask

    # Combine
    total_loss = (pos_loss_in + pos_loss_out + neg_loss_in + neg_loss_out).sum()
    return total_loss / (mask.sum() + 1e-8)


def peak_ordering_loss(peak_probs, targets, mask, fs=25):
    """
    Enforce that positive peak comes before negative peak

    Args:
        peak_probs: (batch, time, 2) - [pos_peak_prob, neg_peak_prob]
        targets: (batch, time) - K-complex labels
        mask: (batch, time) - valid positions
        fs: sampling frequency (after downsampling)

    This loss encourages:
    1. Positive peak to appear first
    2. Negative peak to follow within valid time window (0.08-0.7 seconds)
    """
    B, T = targets.shape
    device = peak_probs.device

    pos_peak = peak_probs[:, :, 0]
    neg_peak = peak_probs[:, :, 1]

    # Time difference constraints (in samples)
    min_diff = int(0.08 * fs)  # minimum 80ms
    max_diff = int(0.7 * fs)   # maximum 700ms

    loss = torch.tensor(0.0, device=device)
    count = 0

    for b in range(B):
        # Find event segments
        event_mask = (targets[b] == 1) & (mask[b] == 1)
        if not event_mask.any():
            continue

        # Get event indices
        event_indices = torch.where(event_mask)[0]
        if len(event_indices) == 0:
            continue

        # Find start and end of events
        diff = torch.cat([
            torch.tensor([1], device=device),
            event_indices[1:] - event_indices[:-1]
        ])
        starts = event_indices[diff > 1]
        if len(starts) == 0:
            starts = event_indices[:1]

        for start_idx in starts:
            # Define event window
            end_idx = start_idx + max_diff
            if end_idx > T:
                end_idx = T

            window = slice(start_idx, end_idx)

            # Get peak probabilities in window
            pos_in_window = pos_peak[b, window]
            neg_in_window = neg_peak[b, window]

            # Find expected peak positions (weighted by probability)
            time_indices = torch.arange(len(pos_in_window), device=device, dtype=torch.float32)
            pos_expected_time = (time_indices * pos_in_window).sum() / (pos_in_window.sum() + 1e-8)
            neg_expected_time = (time_indices * neg_in_window).sum() / (neg_in_window.sum() + 1e-8)

            # Penalty if negative peak comes before positive
            # or if time difference is outside valid range
            time_diff = neg_expected_time - pos_expected_time

            # Loss components
            # 1. Negative should come after positive
            ordering_penalty = F.relu(-time_diff)  # penalty if neg < pos

            # 2. Time difference should be within valid range
            duration_penalty = F.relu(min_diff - time_diff) + F.relu(time_diff - max_diff)

            loss = loss + ordering_penalty + duration_penalty
            count += 1

    return loss / (count + 1e-8)


def zerocrossing_boundary_loss(zerocross_probs, targets, mask):
    """
    Encourage zero-crossing detection at event boundaries

    Args:
        zerocross_probs: (batch, time, 1) - zero-crossing probabilities
        targets: (batch, time) - K-complex labels
        mask: (batch, time) - valid positions

    Logic:
    - At event boundaries (transition from 0→1 or 1→0): high zero-crossing probability
    - Within events or outside: low zero-crossing probability
    """
    B, T = targets.shape

    zerocross = zerocross_probs.squeeze(-1)  # (batch, time)

    # Detect boundaries (transitions)
    padded_targets = F.pad(targets.float(), (1, 1), mode='constant', value=0)
    diff = padded_targets[:, 1:] - padded_targets[:, :-1]
    boundaries = (diff.abs() > 0.5).float()[:, :-1]  # (batch, time)

    # Loss: high probability at boundaries, low elsewhere
    boundary_loss = -torch.log(zerocross + 1e-8) * boundaries * mask
    non_boundary_loss = -torch.log(1 - zerocross + 1e-8) * (1 - boundaries) * mask

    total_loss = (boundary_loss + non_boundary_loss).sum()
    return total_loss / (mask.sum() + 1e-8)


def shape_consistency_loss(raw_signal, predictions, targets, mask, fs=200):
    """
    Validate K-complex shape characteristics in predicted events

    This loss operates on the original signal and validates:
    1. Correct positive→negative peak pattern
    2. Appropriate amplitude
    3. Valid duration

    Args:
        raw_signal: (batch, 1, time_full) - original EEG at full sampling rate
        predictions: (batch, time_pred) - model predictions (0 or 1)
        targets: (batch, time_pred) - ground truth labels
        mask: (batch, time_pred) - valid positions
        fs: original sampling frequency
    """
    device = raw_signal.device
    B = raw_signal.shape[0]

    # Upsample predictions to match raw signal
    stride = raw_signal.shape[2] // predictions.shape[1]

    loss = torch.tensor(0.0, device=device)
    count = 0

    for b in range(B):
        # Find predicted events
        pred_events = predictions[b] > 0.5
        if not pred_events.any():
            continue

        # Get event indices
        event_indices = torch.where(pred_events)[0]
        diff = torch.cat([
            torch.tensor([1], device=device),
            event_indices[1:] - event_indices[:-1]
        ])
        starts = event_indices[diff > 1]
        if len(starts) == 0:
            starts = event_indices[:1]

        # Check each event
        for start_idx in starts:
            # Find event end
            end_search = event_indices[event_indices >= start_idx]
            if len(end_search) == 0:
                continue
            end_idx = end_search[-1] if len(end_search) > 0 else start_idx

            # Map to raw signal indices
            raw_start = int(start_idx.item() * stride)
            raw_end = int((end_idx.item() + 1) * stride)

            if raw_end > raw_signal.shape[2]:
                raw_end = raw_signal.shape[2]

            # Extract segment
            segment = raw_signal[b, 0, raw_start:raw_end]

            if len(segment) < 10:  # too short
                continue

            # Filter signal (0.5-4 Hz for K-complex)
            segment_np = segment.detach().cpu().numpy()
            try:
                nyquist = fs / 2
                b_filter, a_filter = butter(4, [0.5 / nyquist, 4.0 / nyquist], btype='band')
                filtered = filtfilt(b_filter, a_filter, segment_np)
                filtered = torch.from_numpy(filtered).to(device).float()
            except:
                filtered = segment

            # Check for positive and negative peaks
            max_val = filtered.max()
            min_val = filtered.min()
            max_idx = filtered.argmax()
            min_idx = filtered.argmin()

            # Penalties
            # 1. Positive peak should come before negative
            if max_idx >= min_idx:
                loss = loss + 1.0
                count += 1

            # 2. Peak-to-peak amplitude should be reasonable (15-250 µV)
            amplitude = max_val - min_val
            if amplitude < 15:
                loss = loss + (15 - amplitude) / 15
                count += 1
            elif amplitude > 250:
                loss = loss + (amplitude - 250) / 250
                count += 1

            # 3. Duration should be valid (0.08-0.7 seconds)
            duration = len(segment) / fs
            if duration < 0.08:
                loss = loss + (0.08 - duration) / 0.08
                count += 1
            elif duration > 0.7:
                loss = loss + (duration - 0.7) / 0.7
                count += 1

    return loss / (count + 1e-8) if count > 0 else torch.tensor(0.0, device=device)


class KComplexLoss(nn.Module):
    """
    Combined loss for K-complex detection

    This loss combines:
    1. Primary task: K-complex detection (focal loss)
    2. Auxiliary tasks:
       - Peak alignment
       - Peak ordering (pos before neg)
       - Zero-crossing at boundaries
       - Shape consistency
    """

    def __init__(self,
                 alpha_focal=0.25,
                 gamma_focal=2.0,
                 weight_detection=1.0,
                 weight_peak_align=0.3,
                 weight_peak_order=0.2,
                 weight_zerocross=0.2,
                 weight_shape=0.1,
                 fs=200):
        super().__init__()
        self.alpha_focal = alpha_focal
        self.gamma_focal = gamma_focal
        self.weight_detection = weight_detection
        self.weight_peak_align = weight_peak_align
        self.weight_peak_order = weight_peak_order
        self.weight_zerocross = weight_zerocross
        self.weight_shape = weight_shape
        self.fs = fs

    def forward(self, outputs, targets, mask, raw_signal=None):
        """
        Args:
            outputs: dict with keys 'logits', 'peaks', 'zerocross'
                     or just logits tensor (batch, time, 2)
            targets: (batch, time) - ground truth
            mask: (batch, time) - valid positions
            raw_signal: (batch, 1, time_full) - optional, for shape loss

        Returns:
            loss: scalar
            loss_dict: dictionary of individual loss components
        """
        # Handle both dict and tensor inputs
        if isinstance(outputs, dict):
            logits = outputs['logits']
            peaks = outputs.get('peaks', None)
            zerocross = outputs.get('zerocross', None)
        else:
            logits = outputs
            peaks = None
            zerocross = None

        # Primary loss: K-complex detection
        loss_detection = masked_focal_loss(logits, targets, mask,
                                          self.alpha_focal, self.gamma_focal)
        total_loss = self.weight_detection * loss_detection

        loss_dict = {'detection': loss_detection.item()}

        # Auxiliary losses (if available)
        if peaks is not None and self.weight_peak_align > 0:
            loss_peak_align = peak_alignment_loss(peaks, targets, mask)
            total_loss = total_loss + self.weight_peak_align * loss_peak_align
            loss_dict['peak_align'] = loss_peak_align.item()

        if peaks is not None and self.weight_peak_order > 0:
            loss_peak_order = peak_ordering_loss(peaks, targets, mask, fs=self.fs // 8)
            total_loss = total_loss + self.weight_peak_order * loss_peak_order
            loss_dict['peak_order'] = loss_peak_order.item()

        if zerocross is not None and self.weight_zerocross > 0:
            loss_zerocross = zerocrossing_boundary_loss(zerocross, targets, mask)
            total_loss = total_loss + self.weight_zerocross * loss_zerocross
            loss_dict['zerocross'] = loss_zerocross.item()

        # Shape consistency loss (requires raw signal)
        if raw_signal is not None and self.weight_shape > 0:
            # Get predictions from logits
            probs = F.softmax(logits, dim=-1)
            predictions = probs[:, :, 1]  # positive class probability

            loss_shape = shape_consistency_loss(raw_signal, predictions, targets, mask, self.fs)
            total_loss = total_loss + self.weight_shape * loss_shape
            loss_dict['shape'] = loss_shape.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# Example usage
if __name__ == "__main__":
    # Create dummy data
    batch_size = 4
    time_len = 100
    full_time = 4000

    # Model outputs (simulated)
    logits = torch.randn(batch_size, time_len, 2)
    peaks = torch.sigmoid(torch.randn(batch_size, time_len, 2))
    zerocross = torch.sigmoid(torch.randn(batch_size, time_len, 1))

    # Ground truth
    targets = torch.randint(0, 2, (batch_size, time_len))
    mask = torch.ones(batch_size, time_len)

    # Raw signal (optional)
    raw_signal = torch.randn(batch_size, 1, full_time)

    # Create loss function
    criterion = KComplexLoss()

    # Test with all outputs
    outputs = {
        'logits': logits,
        'peaks': peaks,
        'zerocross': zerocross
    }

    loss, loss_dict = criterion(outputs, targets, mask, raw_signal)
    print("Loss with all auxiliary tasks:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # Test with logits only
    loss_simple, loss_dict_simple = criterion(logits, targets, mask)
    print("\nLoss with detection only:")
    for key, value in loss_dict_simple.items():
        print(f"  {key}: {value:.4f}")
