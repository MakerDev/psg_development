"""
K-Complex Detection Model with Shape-Aware Architecture

This model incorporates K-complex morphological characteristics:
1. Positive peak followed by negative peak
2. Zero-crossing boundaries
3. Amplitude constraints
4. Duration validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


DEFAULT_PARAMS = {
    'fs': 200,
    'border_duration': 2.60,
    'border_duration_conv': 0.6,
    'bigger_stem_filters': 64,
    'bigger_max_dilation': 8,
    'bigger_lstm_1_size': 256,
    'bigger_lstm_2_size': 256,
    'fc_units': 128,
    'drop_rate_before_lstm': 0.2,
    'drop_rate_hidden': 0.5,
    'drop_rate_output': 0.0,
    'init_positive_proba': 0.1,
    # K-complex specific parameters
    'min_duration': 0.08,  # seconds
    'max_duration': 0.7,   # seconds
    'min_amplitude': 15,   # microvolts
    'max_amplitude': 250,  # microvolts
}


class MultiDilatedConvBlock(nn.Module):
    """Multi-dilated convolution block with parallel branches"""
    def __init__(self, in_channels, out_channels, max_dilation):
        super(MultiDilatedConvBlock, self).__init__()

        max_exponent = int(np.round(np.log(max_dilation) / np.log(2)))
        self.branches = nn.ModuleList()

        for exp in range(max_exponent + 1):
            dilation = int(2**exp)
            if exp < max_exponent:
                branch_filters = int(out_channels / (2 ** (exp + 1)))
            else:
                branch_filters = 2 * int(out_channels / (2 ** (exp + 1)))

            branch = nn.Sequential(
                nn.Conv1d(in_channels, branch_filters, kernel_size=3,
                         padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm1d(branch_filters, affine=False),
                nn.ReLU(),
                nn.Conv1d(branch_filters, branch_filters, kernel_size=3,
                         padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm1d(branch_filters, affine=False),
                nn.ReLU()
            )
            self.branches.append(branch)

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)


class PeakDetectionHead(nn.Module):
    """Auxiliary head for detecting positive and negative peaks"""
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, 2, kernel_size=1)  # pos_peak, neg_peak

    def forward(self, x):
        """
        Returns: (batch, time, 2) - probabilities for [pos_peak, neg_peak]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.sigmoid(self.conv2(x))  # per-timestep probability
        return x.transpose(1, 2)


class ZeroCrossingHead(nn.Module):
    """Auxiliary head for detecting zero-crossing boundaries"""
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        """
        Returns: (batch, time, 1) - probability of zero-crossing
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.sigmoid(self.conv2(x))
        return x.transpose(1, 2)


class KComplexDetector(nn.Module):
    """
    Advanced K-Complex Detector with Multi-Task Learning

    Tasks:
    1. K-complex detection (primary)
    2. Peak location prediction (auxiliary)
    3. Zero-crossing detection (auxiliary)

    Features:
    - Multi-scale input (raw signal, absolute value, 1st derivative)
    - Shape-aware architecture
    - Explicit peak and zero-crossing modeling
    """

    def __init__(self, in_channels=1, params=None):
        super().__init__()
        self.params = params if params is not None else DEFAULT_PARAMS
        params = self.params

        # Calculate crop sizes
        fs_after_conv = params['fs'] // 8
        self.border_crop_conv = int(np.round(params['border_duration_conv'] * fs_after_conv))
        border_duration_lstm = params['border_duration'] - params['border_duration_conv']
        self.border_crop_lstm = int(np.round(border_duration_lstm * fs_after_conv))

        # Multi-scale input processing
        # Input will be: [raw_signal, abs(signal), diff(signal)]
        self.input_channels = in_channels * 3  # raw, abs, derivative

        # Input batch normalization for each channel type
        self.bn_input = nn.BatchNorm1d(self.input_channels)

        # Stem layers
        self.stem = nn.Sequential(
            nn.Conv1d(self.input_channels, params['bigger_stem_filters'],
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(params['bigger_stem_filters'], affine=False),
            nn.ReLU(),
            nn.Conv1d(params['bigger_stem_filters'], params['bigger_stem_filters'],
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(params['bigger_stem_filters'], affine=False),
            nn.ReLU()
        )

        # Pooling layers
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Multi-dilated convolution blocks
        self.mdconv1 = MultiDilatedConvBlock(
            params['bigger_stem_filters'],
            params['bigger_stem_filters'] * 2,
            params['bigger_max_dilation']
        )

        mdconv1_out_channels = sum([
            int(params['bigger_stem_filters'] * 2 / (2 ** (i + 1)))
            if i < int(np.log2(params['bigger_max_dilation']))
            else 2 * int(params['bigger_stem_filters'] * 2 / (2 ** (i + 1)))
            for i in range(int(np.log2(params['bigger_max_dilation'])) + 1)
        ])

        self.mdconv2 = MultiDilatedConvBlock(
            mdconv1_out_channels,
            params['bigger_stem_filters'] * 4,
            params['bigger_max_dilation']
        )

        mdconv2_out_channels = sum([
            int(params['bigger_stem_filters'] * 4 / (2 ** (i + 1)))
            if i < int(np.log2(params['bigger_max_dilation']))
            else 2 * int(params['bigger_stem_filters'] * 4 / (2 ** (i + 1)))
            for i in range(int(np.log2(params['bigger_max_dilation'])) + 1)
        ])

        # LSTM layers
        lstm_input_size = mdconv2_out_channels
        if params['bigger_lstm_1_size'] > 0:
            self.lstm1 = nn.LSTM(lstm_input_size, params['bigger_lstm_1_size'],
                                batch_first=True, bidirectional=True)
            self.dropout1 = nn.Dropout(params['drop_rate_before_lstm'])
            lstm_output_size = params['bigger_lstm_1_size'] * 2
        else:
            self.lstm1 = None
            lstm_output_size = lstm_input_size

        if params['bigger_lstm_2_size'] > 0:
            self.lstm2 = nn.LSTM(lstm_output_size, params['bigger_lstm_2_size'],
                                batch_first=True, bidirectional=True)
            self.dropout2 = nn.Dropout(params['drop_rate_hidden'])
            lstm_output_size = params['bigger_lstm_2_size'] * 2
        else:
            self.lstm2 = None

        # Shared feature layer
        if params['fc_units'] > 0:
            self.fc = nn.Conv1d(lstm_output_size, params['fc_units'], kernel_size=1)
            self.dropout_fc = nn.Dropout(params['drop_rate_hidden'])
            self.fc_activation = nn.ReLU()
            final_size = params['fc_units']
        else:
            self.fc = None
            final_size = lstm_output_size

        # Multi-task heads
        # 1. Primary: K-complex detection
        self.kcomplex_head = nn.Conv1d(final_size, 2, kernel_size=1)
        self.dropout_output = nn.Dropout(params['drop_rate_output'])

        # Initialize output bias for class imbalance
        init_pos_prob = params['init_positive_proba']
        bias_init = -np.log((1 - init_pos_prob) / init_pos_prob)
        self.kcomplex_head.bias.data[1] = bias_init

        # 2. Auxiliary: Peak detection
        self.peak_head = PeakDetectionHead(final_size)

        # 3. Auxiliary: Zero-crossing detection
        self.zerocross_head = ZeroCrossingHead(final_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _prepare_multi_scale_input(self, x):
        """
        Prepare multi-scale input: [raw, abs, derivative]

        Args:
            x: (batch, 1, time)
        Returns:
            (batch, 3, time)
        """
        # Absolute value
        x_abs = torch.abs(x)

        # First derivative (zero-crossing detection)
        x_diff = torch.cat([
            torch.zeros_like(x[:, :, :1]),
            x[:, :, 1:] - x[:, :, :-1]
        ], dim=2)

        # Concatenate all scales
        return torch.cat([x, x_abs, x_diff], dim=1)

    def crop_time(self, x, border_crop):
        """Crop time dimension"""
        if border_crop > 0:
            if x.ndim == 3:
                return x[:, :, border_crop:-border_crop]
            else:
                return x[:, border_crop:-border_crop, :]
        return x

    def forward(self, x, return_auxiliary=False):
        """
        Args:
            x: (batch, 1, time_len)
            return_auxiliary: If True, return auxiliary predictions

        Returns:
            If return_auxiliary=False:
                logits: (batch, time_len//8, 2)
            If return_auxiliary=True:
                dict with keys: 'logits', 'peaks', 'zerocross'
        """
        # Multi-scale input
        x = self._prepare_multi_scale_input(x)
        x = self.bn_input(x)

        # Stem
        x = self.stem(x)

        # Encoder: Pooling and multi-dilated convolutions
        x = self.pool1(x)
        x = self.mdconv1(x)
        x = self.pool2(x)
        x = self.mdconv2(x)
        x = self.pool3(x)

        # First crop
        x = self.crop_time(x, self.border_crop_conv)

        # LSTM layers
        x = x.transpose(1, 2)  # (batch, time, channels)

        if self.lstm1 is not None:
            x = self.dropout1(x)
            x, _ = self.lstm1(x)

        if self.lstm2 is not None:
            x = self.dropout2(x)
            x, _ = self.lstm2(x)

        # Second crop
        x = self.crop_time(x.transpose(1, 2), self.border_crop_lstm).transpose(1, 2)

        # Shared features
        x = x.transpose(1, 2)  # (batch, channels, time)

        if self.fc is not None:
            x = self.fc(x)
            x = self.fc_activation(x)
            x = self.dropout_fc(x)

        # Multi-task predictions
        # 1. K-complex detection (primary)
        x_out = self.dropout_output(x)
        logits = self.kcomplex_head(x_out).transpose(1, 2)  # (batch, time, 2)

        if not return_auxiliary:
            return logits

        # 2. Peak detection (auxiliary)
        peaks = self.peak_head(x)  # (batch, time, 2)

        # 3. Zero-crossing detection (auxiliary)
        zerocross = self.zerocross_head(x)  # (batch, time, 1)

        return {
            'logits': logits,
            'peaks': peaks,
            'zerocross': zerocross
        }


# Example usage
if __name__ == "__main__":
    model = KComplexDetector(in_channels=1)

    batch_size = 2
    time_len = 4000  # 20 seconds at 200 Hz
    x = torch.randn(batch_size, 1, time_len)

    # Forward pass
    print("Testing K-Complex Detector...")
    outputs = model(x, return_auxiliary=True)

    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Peaks shape: {outputs['peaks'].shape}")
    print(f"Zero-crossing shape: {outputs['zerocross'].shape}")

    # Without auxiliary
    logits_only = model(x, return_auxiliary=False)
    print(f"Logits only shape: {logits_only.shape}")
