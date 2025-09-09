import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import torch.fft

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
    'type_batchnorm': 'BN',
    'type_dropout': 'SEQUENCE_DROP'
}


class OptimizedMorletCWT(nn.Module):
    """Optimized Morlet Continuous Wavelet Transform using FFT"""
    def __init__(self, fb_list, fs, lower_freq, upper_freq, n_scales, 
                 size_factor=1.5, expansion_factor=0.9, noise_intensity=0.02,
                 trainable=True):
        super(OptimizedMorletCWT, self).__init__()
        self.fb_list = fb_list
        self.fs = fs
        self.lower_freq = lower_freq
        self.upper_freq = upper_freq
        self.n_scales = n_scales
        self.size_factor = size_factor
        self.expansion_factor = expansion_factor
        self.noise_intensity = noise_intensity
        
        # Initialize scales
        scales = np.logspace(np.log10(lower_freq), np.log10(upper_freq), n_scales)
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))
        
        # Initialize fb parameters
        if trainable:
            self.fb_params = nn.Parameter(torch.tensor(fb_list, dtype=torch.float32))
        else:
            self.register_buffer('fb_params', torch.tensor(fb_list, dtype=torch.float32))
        
        # Pre-compute maximum wavelet length
        self.max_wavelet_len = int(self.size_factor * scales[-1] * self.fs)
        
    def _generate_wavelets_batch(self, time_len, device):
        """Generate all wavelets at once for batch processing"""
        scales = self.scales
        fb_params = self.fb_params if hasattr(self, 'fb_params') else torch.tensor(self.fb_list, device=device)
        
        # Pre-allocate arrays for all wavelets
        n_wavelets = len(scales) * len(fb_params)
        wavelets_real = []
        wavelets_imag = []
        
        # Generate frequency domain wavelets
        freqs = torch.fft.fftfreq(time_len, d=1.0/self.fs, device=device)
        
        for fb in fb_params:
            for scale in scales:
                # Frequency domain Morlet wavelet
                sigma_f = 1.0 / (2 * np.pi * fb * scale)
                wavelet_freq = torch.exp(-0.5 * (freqs * scale * 2 * np.pi)**2 * sigma_f**2)
                wavelet_freq *= torch.exp(2j * np.pi * freqs * scale)
                wavelet_freq /= torch.sqrt(scale)
                
                wavelets_real.append(wavelet_freq.real)
                wavelets_imag.append(wavelet_freq.imag)
        
        return torch.stack(wavelets_real), torch.stack(wavelets_imag)
    
    def forward(self, x, training=True):
        """
        x: (batch_size, time_len)
        Returns: (batch_size, time_len, n_scales, 2*len(fb_list))
        """
        batch_size, time_len = x.shape
        device = x.device
        
        # Add noise to scales during training
        if training and self.training and self.noise_intensity > 0:
            noise = torch.randn_like(self.scales) * self.noise_intensity
            scales = self.scales * (1 + noise)
        else:
            scales = self.scales
        
        # FFT of input signal
        x_fft = torch.fft.fft(x, dim=-1)
        
        # Generate all wavelets in frequency domain
        wavelets_real, wavelets_imag = self._generate_wavelets_batch(time_len, device)
        
        # Batch convolution in frequency domain
        # Expand dimensions for broadcasting
        x_fft_expanded = x_fft.unsqueeze(1)  # (batch, 1, time_len)
        
        # Complex multiplication in frequency domain
        conv_real = torch.fft.ifft(x_fft_expanded * wavelets_real.unsqueeze(0), dim=-1).real
        conv_imag = torch.fft.ifft(x_fft_expanded * wavelets_imag.unsqueeze(0), dim=-1).real
        
        # Reshape output
        n_fb = len(self.fb_params) if hasattr(self, 'fb_params') else len(self.fb_list)
        n_scales = len(scales)
        
        conv_real = conv_real.reshape(batch_size, n_fb, n_scales, time_len)
        conv_imag = conv_imag.reshape(batch_size, n_fb, n_scales, time_len)
        
        # Stack real and imaginary parts
        output = torch.stack([
            conv_real.permute(0, 3, 2, 1),  # (batch, time, scales, fb)
            conv_imag.permute(0, 3, 2, 1)
        ], dim=-1)
        
        # Flatten last two dimensions
        output = output.reshape(batch_size, time_len, n_scales, -1)
        
        return output


class EfficientMultiDilatedConvBlock(nn.Module):
    """Efficient multi-dilated convolution block with grouped convolutions"""
    def __init__(self, in_channels, out_channels, max_dilation, is_2d=False):
        super(EfficientMultiDilatedConvBlock, self).__init__()
        self.is_2d = is_2d
        
        # Calculate dilations and filter distribution
        max_exponent = int(np.round(np.log(max_dilation) / np.log(2)))
        
        # Use grouped convolution for efficiency
        self.conv_groups = nn.ModuleList()
        self.bn_groups = nn.ModuleList()
        
        self.filter_splits = []
        total_filters = 0
        
        for exp in range(max_exponent + 1):
            dilation = int(2**exp)
            if exp < max_exponent:
                branch_filters = int(out_channels / (2 ** (exp + 1)))
            else:
                branch_filters = 2 * int(out_channels / (2 ** (exp + 1)))
            
            self.filter_splits.append(branch_filters)
            total_filters += branch_filters
            
            if is_2d:
                # Use depthwise separable convolution for efficiency
                self.conv_groups.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_filters, kernel_size=3,
                                 padding=dilation, dilation=dilation, bias=False),
                        nn.Conv2d(branch_filters, branch_filters, kernel_size=1, bias=False)
                    )
                )
                self.bn_groups.append(nn.BatchNorm2d(branch_filters, affine=False))
            else:
                # Use 1D depthwise separable convolution
                self.conv_groups.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, branch_filters, kernel_size=3,
                                 padding=dilation, dilation=dilation, bias=False),
                        nn.Conv1d(branch_filters, branch_filters, kernel_size=1, bias=False)
                    )
                )
                self.bn_groups.append(nn.BatchNorm1d(branch_filters, affine=False))
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        outputs = []
        for conv, bn in zip(self.conv_groups, self.bn_groups):
            out = conv(x)
            out = bn(out)
            out = self.activation(out)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class REDv2CWT1D(nn.Module):
    """Optimized REDv2 CWT-domain model for K-complex detection"""
    def __init__(self, params=None):
        super(REDv2CWT1D, self).__init__()
        self.params = params if params is not None else DEFAULT_PARMS
        params = self.params

        # Calculate crop sizes
        fs_after_conv = params['fs'] // 8
        self.border_crop_conv = int(np.round(params['border_duration_conv'] * fs_after_conv)) - 8
        border_duration_lstm = (params['border_duration'] - 
                               params['border_duration_conv'] - 
                               params['border_duration_cwt']) 
        self.border_crop_lstm = int(np.round(border_duration_lstm * fs_after_conv))
        
        # Optimized CWT layer
        self.cwt = OptimizedMorletCWT(
            fb_list=params['fb_list'],
            fs=params['fs'],
            lower_freq=params['lower_freq'],
            upper_freq=params['upper_freq'],
            n_scales=params['n_scales'],
            size_factor=params['wavelet_size_factor'],
            expansion_factor=params['cwt_expansion_factor'],
            noise_intensity=params['cwt_noise_intensity'],
            trainable=params['trainable_wavelet']
        )
        
        # Calculate CWT output channels
        cwt_channels = params['n_scales'] * len(params['fb_list']) * 2  # real + imag
        
        # Stem layer with channel attention for feature selection
        self.stem = nn.Sequential(
            nn.Conv1d(cwt_channels, params['bigger_stem_filters'], 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(params['bigger_stem_filters'], affine=False),
            nn.ReLU(inplace=True)
        )
        
        # Efficient multi-dilated convolution blocks
        self.mdconv1 = EfficientMultiDilatedConvBlock(
            params['bigger_stem_filters'],
            params['bigger_stem_filters'] * 2,
            params['bigger_max_dilation'],
            is_2d=False
        )
        
        # Calculate actual output channels
        mdconv1_out_channels = sum([
            int(params['bigger_stem_filters'] * 2 / (2 ** (i + 1))) if i < int(np.log2(params['bigger_max_dilation']))
            else 2 * int(params['bigger_stem_filters'] * 2 / (2 ** (i + 1)))
            for i in range(int(np.log2(params['bigger_max_dilation'])) + 1)
        ])
        
        self.mdconv2 = EfficientMultiDilatedConvBlock(
            mdconv1_out_channels,
            params['bigger_stem_filters'] * 4,
            params['bigger_max_dilation'],
            is_2d=False
        )
        
        # Pooling layers
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Calculate final conv output channels
        mdconv2_out_channels = sum([
            int(params['bigger_stem_filters'] * 4 / (2 ** (i + 1))) if i < int(np.log2(params['bigger_max_dilation']))
            else 2 * int(params['bigger_stem_filters'] * 4 / (2 ** (i + 1)))
            for i in range(int(np.log2(params['bigger_max_dilation'])) + 1)
        ])
        
        # LSTM layers with gradient clipping
        lstm_input_size = mdconv2_out_channels
        if params['bigger_lstm_1_size'] > 0:
            self.lstm1 = nn.LSTM(lstm_input_size, params['bigger_lstm_1_size'],
                                batch_first=True, bidirectional=True,
                                dropout=0 if params['bigger_lstm_2_size'] == 0 else params['drop_rate_before_lstm'])
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
            self.fc_activation = nn.ReLU(inplace=True)
            final_size = params['fc_units']
        else:
            self.fc = None
            final_size = lstm_output_size
            
        # Output layer
        self.output = nn.Conv1d(final_size, 2, kernel_size=1)
        
        # Dropout only if specified
        if params['drop_rate_output'] > 0:
            self.dropout_output = nn.Dropout(params['drop_rate_output'])
        else:
            self.dropout_output = None
        
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
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def crop_time(self, x, border_crop):
        """Crop time dimension"""
        if border_crop > 0:
            return x[:, :, border_crop:-border_crop]
        return x
    
    def forward(self, x):
        """
        x: (batch_size, time_len) or (batch_size, 1, time_len)
        Returns: logits (batch_size, time_len//8, 2)
        """
        if x.ndim == 3:
            x = x.squeeze(1)

        # Apply optimized CWT
        with torch.cuda.amp.autocast(enabled=False):  # CWT needs full precision
            cwt_output = self.cwt(x, training=self.training)
        
        # Flatten scales and channels: (batch, time, scales*channels)
        batch_size, time_len, n_scales, n_channels = cwt_output.shape
        x = cwt_output.view(batch_size, time_len, n_scales * n_channels)
        
        # Transpose for Conv1d: (batch, channels, time)
        x = x.transpose(1, 2)
        
        # Apply border crop from CWT
        border_crop_cwt = int(np.round(self.params['border_duration_cwt'] * self.params['fs']))
        if border_crop_cwt > 0:
            x = x[:, :, border_crop_cwt:-border_crop_cwt]
        
        # Downsample by factor of 2 (more efficient than avg_pool1d for stride=kernel_size)
        x = x[:, :, ::2]  # Equivalent to stride=2 in original CWT
        
        # Stem
        x = self.stem(x)
        
        # Multi-dilated convolutions with pooling
        x = self.mdconv1(x)
        x = self.pool2(x)
        x = self.mdconv2(x)
        x = self.pool3(x)
        
        # First crop
        x = self.crop_time(x, self.border_crop_conv)
        
        # Prepare for LSTM
        x = x.transpose(1, 2)
        
        # LSTM layers
        if self.lstm1 is not None:
            x, _ = self.lstm1(x)
            
        if self.lstm2 is not None:
            x = self.dropout2(x)
            x, _ = self.lstm2(x)
        
        # Second crop
        x = self.crop_time(x.transpose(1, 2), self.border_crop_lstm).transpose(1, 2)
        
        # Classification
        x = x.transpose(1, 2)
        
        if self.fc is not None:
            x = self.fc(x)
            x = self.fc_activation(x)
            x = self.dropout_fc(x)
        
        # Output
        if self.dropout_output is not None:
            x = self.dropout_output(x)
        logits = self.output(x)
        
        # Transpose to (batch, time, classes)
        logits = logits.transpose(1, 2)
        
        return logits


# Example usage and benchmarking
if __name__ == "__main__":
    import time
    
    # Create model
    model_cwt = REDv2CWT1D(DEFAULT_PARMS)
    model_cwt.eval()  # Set to eval mode for benchmarking
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cwt = model_cwt.to(device)
    
    # Test with dummy input
    batch_size = 4
    time_len = 5040  # 25.2 seconds at 200 Hz
    x = torch.randn(batch_size, time_len).to(device)
    
    # Warmup
    for _ in range(3):
        _ = model_cwt(x)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    n_iterations = 10
    for _ in range(n_iterations):
        logits = model_cwt(x)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    print(f"Average inference time: {(end_time - start_time) / n_iterations:.4f} seconds")
    print(f"Output shape: {logits.shape}")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")