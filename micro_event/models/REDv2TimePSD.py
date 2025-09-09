# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import datetime as dt
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal

# Note: This implementation requires the 'geoopt' library for the full Riemannian
# version, but this simplified implementation does not strictly require it.
# It is good practice to have it installed if extending the functionality.
# pip install geoopt
import geoopt

# -----------------------------------------------------------------------------
# Constants & Default Parameters
# -----------------------------------------------------------------------------
DEFAULT_PARMS = {
    'fs': 200,
    'border_duration': 2.60,
    'border_duration_conv': 0.6,
    'border_duration_cwt': 2.31,
    'bigger_stem_filters': 64,
    'bigger_max_dilation': 8,
    'bigger_lstm_1_size': 256,
    'bigger_lstm_2_size': 256,
    'fc_units': 128,
    'drop_rate_before_lstm': 0.2,
    'drop_rate_hidden': 0.5,
    'drop_rate_output': 0.0,
    'init_positive_proba': 0.1,
    'fb_list': [0.1323],
    'trainable_wavelet': True,
    'wavelet_size_factor': 1.5,
    'lower_freq': 0.5,
    'upper_freq': 30,
    'n_scales': 32,
    'cwt_noise_intensity': 0.02,
    'cwt_expansion_factor': 0.9,
    'cwt_return_real_part': True,
    'cwt_return_imag_part': True,
    'cwt_return_magnitude': False,
    'cwt_return_phase': False,
    'type_batchnorm': 'PSD', # Can be 'BN' or 'PSD'
    'type_dropout': 'SEQUENCE_DROP'
}

# -----------------------------------------------------------------------------
# PSDNorm Layer Implementation (Corrected)
# -----------------------------------------------------------------------------

class PSDNorm(nn.Module):
    """
    PSDNorm: A temporal normalization layer for deep learning models.
    This layer normalizes the Power Spectral Density (PSD) of feature maps.
    This is a simplified implementation that normalizes the power spectrum vector.
    """
    def __init__(self, num_features, eps=1e-6, momentum=0.1, n_fft=128):
        super(PSDNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.n_fft = n_fft
        
        # The length of the real FFT output
        self.fft_len = n_fft // 2 + 1

        # Correctly initialize running_barycenter as a vector of mean powers
        self.register_buffer('running_barycenter', torch.ones(1, num_features, self.fft_len))
        self.running_barycenter.requires_grad = False

    def forward(self, x):
        # x shape: (batch_size, num_features, time_len)
        
        # 1. Estimate Power Spectrum
        X_fft = torch.fft.rfft(x, n=self.n_fft, dim=-1)
        power_spectrum = torch.abs(X_fft)**2

        if self.training:
            # Calculate batch mean power spectrum across the batch dimension
            batch_mean_power = torch.mean(power_spectrum, dim=0, keepdim=True)
            
            # Update running barycenter (mean power)
            self.running_barycenter = (1 - self.momentum) * self.running_barycenter + self.momentum * batch_mean_power.detach()
            
            current_mean = batch_mean_power
        else:
            current_mean = self.running_barycenter

        # 2. Normalize
        # Normalize the FFT coefficients by the square root of the mean power
        # Broadcasting rules will apply current_mean across the batch dimension
        normalized_fft = X_fft / (torch.sqrt(current_mean) + self.eps)
        
        # 3. Inverse FFT to get normalized signal, ensuring original length
        y = torch.fft.irfft(normalized_fft, n=x.size(-1), dim=-1)
        
        return y

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

def get_norm_layer(num_features, params):
    """Helper function to select normalization layer."""
    if params.get('type_batchnorm', 'BN') == 'PSD':
        # Using a simplified print statement to avoid excessive output
        # print(f"Using PSDNorm for {num_features} features.")
        return PSDNorm(num_features)
    else:
        return nn.BatchNorm1d(num_features, affine=False)

class MultiDilatedConvBlock(nn.Module):
    """Multi-dilated convolution block with parallel branches"""
    def __init__(self, in_channels, out_channels, max_dilation, params, is_2d=False):
        super(MultiDilatedConvBlock, self).__init__()
        self.is_2d = is_2d
        
        # Calculate dilations
        max_exponent = int(np.round(np.log(max_dilation) / np.log(2)))
        self.branches = nn.ModuleList()
        
        total_filters = 0
        for exp in range(max_exponent + 1):
            dilation = int(2**exp)
            if exp < max_exponent:
                branch_filters = int(out_channels / (2 ** (exp + 1)))
            else:
                branch_filters = 2 * int(out_channels / (2 ** (exp + 1)))
            total_filters += branch_filters
            
            if is_2d:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, branch_filters, kernel_size=3, 
                              padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(branch_filters, affine=False),
                    nn.ReLU(),
                    nn.Conv2d(branch_filters, branch_filters, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(branch_filters, affine=False),
                    nn.ReLU()
                )
            else:
                branch = nn.Sequential(
                    nn.Conv1d(in_channels, branch_filters, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False),
                    get_norm_layer(branch_filters, params),
                    nn.ReLU(),
                    nn.Conv1d(branch_filters, branch_filters, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False),
                    get_norm_layer(branch_filters, params),
                    nn.ReLU()
                )
            self.branches.append(branch)
    
    def forward(self, x):
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))
        return torch.cat(outputs, dim=1)


class REDv2TimePSD(nn.Module):
    """REDv2 Time-domain model for K-complex detection"""
    def __init__(self, in_channels=1, params=None):
        super(REDv2TimePSD, self).__init__()
        self.params = params if params is not None else DEFAULT_PARMS
        params = self.params
        
        # Calculate crop sizes
        fs_after_conv = params['fs'] // 8
        self.border_crop_conv = int(np.round(params['border_duration_conv'] * fs_after_conv))
        border_duration_lstm = params['border_duration'] - params['border_duration_conv']
        self.border_crop_lstm = int(np.round(border_duration_lstm * fs_after_conv))
        
        # Input batch normalization
        self.bn_input = get_norm_layer(in_channels, params)
        
        # Stem layers
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, params['bigger_stem_filters'], kernel_size=3, padding=1, bias=False),
            get_norm_layer(params['bigger_stem_filters'], params),
            nn.ReLU(),
            nn.Conv1d(params['bigger_stem_filters'], params['bigger_stem_filters'], 
                      kernel_size=3, padding=1, bias=False),
            get_norm_layer(params['bigger_stem_filters'], params),
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
            params['bigger_max_dilation'],
            params,
            is_2d=False
        )
        
        # Calculate actual output channels from mdconv1
        mdconv1_out_channels = sum([
            int(params['bigger_stem_filters'] * 2 / (2 ** (i + 1))) if i < int(np.log2(params['bigger_max_dilation']))
            else 2 * int(params['bigger_stem_filters'] * 2 / (2 ** (i + 1)))
            for i in range(int(np.log2(params['bigger_max_dilation'])) + 1)
        ])
        
        self.mdconv2 = MultiDilatedConvBlock(
            mdconv1_out_channels,
            params['bigger_stem_filters'] * 4,
            params['bigger_max_dilation'],
            params,
            is_2d=False
        )
        
        # Calculate actual output channels from mdconv2
        mdconv2_out_channels = sum([
            int(params['bigger_stem_filters'] * 4 / (2 ** (i + 1))) if i < int(np.log2(params['bigger_max_dilation']))
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
        
        # Classification layers
        if params['fc_units'] > 0:
            self.fc = nn.Conv1d(lstm_output_size, params['fc_units'], kernel_size=1)
            self.dropout_fc = nn.Dropout(params['drop_rate_hidden'])
            self.fc_activation = nn.ReLU()
            final_size = params['fc_units']
        else:
            self.fc = None
            final_size = lstm_output_size
            
        # Output layer with initialization for positive probability
        self.output = nn.Conv1d(final_size, 2, kernel_size=1)
        self.dropout_output = nn.Dropout(params['drop_rate_output'])
        
        # Initialize output bias
        init_pos_prob = params['init_positive_proba']
        bias_init = -np.log((1 - init_pos_prob) / init_pos_prob)
        self.output.bias.data[1] = bias_init
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def crop_time(self, x, border_crop):
        """Crop time dimension"""
        if border_crop > 0:
            return x[..., border_crop:-border_crop]
        return x
    
    def forward(self, x):
        """
        x: (batch_size, in_channels, time_len)
        Returns: logits (batch_size, time_len//8, 2)
        """
        # Input normalization
        x = self.bn_input(x)
        
        # Stem
        x = self.stem(x)
        
        # Pooling and multi-dilated convolutions
        x = self.pool1(x)
        x = self.mdconv1(x)
        x = self.pool2(x)
        x = self.mdconv2(x)
        x = self.pool3(x)
        
        # First crop
        x = self.crop_time(x, self.border_crop_conv)
        
        # Prepare for LSTM: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        
        # LSTM layers
        if self.lstm1 is not None:
            x = self.dropout1(x)
            x, _ = self.lstm1(x)
            
        if self.lstm2 is not None:
            x = self.dropout2(x)
            x, _ = self.lstm2(x)
        
        # Second crop
        x = self.crop_time(x.transpose(1, 2), self.border_crop_lstm).transpose(1, 2)
        
        # Classification
        x = x.transpose(1, 2)  # Back to (batch, channels, time)
        
        if self.fc is not None:
            x = self.fc(x)
            x = self.fc_activation(x)
            x = self.dropout_fc(x)
        
        # Output
        x = self.dropout_output(x)
        logits = self.output(x)
        
        # Transpose to (batch, time, classes)
        logits = logits.transpose(1, 2)
        
        return logits


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Test with standard BatchNorm ---
    print("--- Testing with BatchNorm1d ---")
    params_bn = DEFAULT_PARMS.copy()
    params_bn['type_batchnorm'] = 'BN'
    
    model_bn = REDv2TimePSD(in_channels=1, params=params_bn)
    
    batch_size = 2
    time_len = 5040  # Example: 25.2 seconds at 200 Hz
    # Corrected input shape to (batch, channels, time)
    x_input = torch.randn(batch_size, 1, time_len)

    logits_bn = model_bn(x_input)
    print(f"BatchNorm model output shape: {logits_bn.shape}")
    print("-" * 30, "\n")

    # --- Test with PSDNorm ---
    print("--- Testing with PSDNorm ---")
    params_psd = DEFAULT_PARMS.copy()
    params_psd['type_batchnorm'] = 'PSD'
    
    model_psd = REDv2TimePSD(in_channels=1, params=params_psd)
    
    logits_psd = model_psd(x_input)
    print(f"PSDNorm model output shape: {logits_psd.shape}")
    print("-" * 30)

    # Verify that the model runs and produces the correct output shape
    expected_time_len = (time_len // 8) - (model_psd.border_crop_conv + model_psd.border_crop_lstm)
    print(f"Input time length: {time_len}")
    print(f"Expected output time length: {expected_time_len}")
    assert logits_psd.shape == (batch_size, expected_time_len, 2)
    print("Test passed!")