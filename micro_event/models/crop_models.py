import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import scipy.signal

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
    

# class MorletCWT(nn.Module):
#     """Morlet Continuous Wavelet Transform layer for PyTorch"""
#     def __init__(self, fb_list, fs, lower_freq, upper_freq, n_scales, 
#                  size_factor=1.5, expansion_factor=0.9, noise_intensity=0.02,
#                  trainable=True):
#         super(MorletCWT, self).__init__()
#         self.fb_list = fb_list
#         self.fs = fs
#         self.lower_freq = lower_freq
#         self.upper_freq = upper_freq
#         self.n_scales = n_scales
#         self.size_factor = size_factor
#         self.expansion_factor = expansion_factor
#         self.noise_intensity = noise_intensity
        
#         # Initialize scales
#         scales = np.logspace(np.log10(lower_freq), np.log10(upper_freq), n_scales)
#         self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))
        
#         # Initialize fb parameters
#         if trainable:
#             self.fb_params = nn.Parameter(torch.tensor(fb_list, dtype=torch.float32))
#         else:
#             self.register_buffer('fb_params', torch.tensor(fb_list, dtype=torch.float32))
    
#     def forward(self, x, training=True):
#         """
#         x: (batch_size, time_len)
#         Returns: (batch_size, time_len, n_scales, 2*len(fb_list))
#         """
#         batch_size, time_len = x.shape
#         device = x.device
        
#         # Add noise during training
#         if training and self.noise_intensity > 0:
#             noise = torch.randn_like(self.scales) * self.noise_intensity
#             scales = self.scales * (1 + noise)
#         else:
#             scales = self.scales
        
#         # Compute wavelets for each scale and fb
#         outputs = []
#         for fb in (self.fb_params if hasattr(self, 'fb_params') else self.fb_list):
#             for scale in scales:
#                 # Generate Morlet wavelet
#                 # This is a simplified version - for production use pytorch-wavelets
#                 wavelet_len = int(self.size_factor * scale * self.fs)
#                 t = torch.arange(-wavelet_len//2, wavelet_len//2, device=device) / self.fs
                
#                 # Complex Morlet wavelet
#                 sigma = fb * scale
#                 wavelet = torch.exp(2j * np.pi * t / scale) * torch.exp(-t**2 / (2 * sigma**2))
#                 wavelet = wavelet / torch.sqrt(scale)
                
#                 # Convolve with signal
#                 # For simplicity, using 1D conv - in practice, use FFT for efficiency
#                 x_padded = F.pad(x, (wavelet_len//2, wavelet_len//2), mode='reflect')
                
#                 # Real and imaginary parts
#                 real_conv = F.conv1d(x_padded.unsqueeze(1), 
#                                     wavelet.real.unsqueeze(0).unsqueeze(0), 
#                                     padding=0)
#                 imag_conv = F.conv1d(x_padded.unsqueeze(1), 
#                                     wavelet.imag.unsqueeze(0).unsqueeze(0), 
#                                     padding=0)
                
#                 if real_conv.shape[2] != time_len:
#                     real_conv = real_conv[:, :, :time_len]
#                     imag_conv = imag_conv[:, :, :time_len]

#                 outputs.append(torch.stack([real_conv.squeeze(1), imag_conv.squeeze(1)], dim=-1))
        
#         # Stack all outputs
#         output = torch.stack(outputs, dim=2)
#         output = output.permute(0, 1, 2, 3).contiguous()
        
#         return output
class MorletCWT(nn.Module):
    """
    Fast Morlet-CWT layer (1-D)  
    – 모든 스케일·fb 에 대한 실수/허수 필터를 한 번의 conv1d 호출로 계산
    """
    def __init__(
        self,
        fb_list,
        fs: int,
        lower_freq: float,
        upper_freq: float,
        n_scales: int,
        size_factor: float = 1.5,
        noise_intensity: float = 0.02,
        trainable: bool = True,
    ):
        super().__init__()

        self.fs = fs
        self.noise_intensity = noise_intensity
        self.trainable_fb = trainable

        # 스케일(=주파수 상관 길이) 정의 – logspace
        scales = torch.logspace(
            np.log10(lower_freq),
            np.log10(upper_freq),
            n_scales,
            dtype=torch.float32
        )
        self.register_buffer("scales", scales)                 # (S,)

        # fb (중심주파수/σ) 파라미터
        fb = torch.as_tensor(fb_list, dtype=torch.float32)      # (F,)
        if trainable:
            self.fb_params = nn.Parameter(fb)
        else:
            self.register_buffer("fb_params", fb)

        # 커널 길이 (가장 긴 스케일 기준으로 고정)
        self.max_len = int(size_factor * scales.max().item() * fs)
        if self.max_len % 2 == 0:                   # 홀수 길이로 맞추면 중앙 정렬 편리
            self.max_len += 1
        self.padding = self.max_len // 2            # 'same' 패딩

        # 시간축 텐서 (1, 1, K) – conv 호출 직전 device로 이동
        t = torch.arange(-(self.max_len // 2),
                         self.max_len // 2 + 1,
                         dtype=torch.float32) / fs      # (K,)
        self.register_buffer("t_long", t)

    # --------------------------------------------------
    # 내부 함수: wavelet bank (실수·허수 필터) 생성
    # --------------------------------------------------
    def _build_wavelet_bank(self, device: torch.device, training: bool):
        """
        Returns
        -------
        weight : torch.Tensor
            shape (2 * F * S, 1, K)
        """
        scales = self.scales.to(device)                          # (S,)
        fb = (self.fb_params if self.trainable_fb else
              getattr(self, "fb_params")).to(device)             # (F,)

        if training and (self.noise_intensity > 0):
            noise = (1 + torch.randn_like(scales) * self.noise_intensity)
            scales = scales * noise

        S, F = scales.numel(), fb.numel()
        K = self.max_len
        t = self.t_long.to(device)                               # (K,)

        # broadcasting: (F, S, K)
        t_grid = t.view(1, 1, K)
        scale_grid = scales.view(1, S, 1)
        fb_grid = fb.view(F, 1, 1)

        sigma = fb_grid * scale_grid                             # (F, S, 1)
        exponent = 2j * np.pi * t_grid / scale_grid             # (F, S, K)
        gaussian = torch.exp(-t_grid.pow(2) / (2 * sigma.pow(2)))
        wavelet = torch.exp(exponent) * gaussian / scale_grid.sqrt()

        # 실수/허수 분리 → (2*F*S, 1, K)
        real_k = wavelet.real.reshape(-1, K)
        imag_k = wavelet.imag.reshape(-1, K)
        kernels = torch.cat([real_k, imag_k], dim=0).unsqueeze(1)

        return kernels.contiguous()          # (2FS,1,K)

    # --------------------------------------------------
    def forward(self, x: torch.Tensor, training: bool = True):
        """
        Parameters
        ----------
        x : (B, T)  or (B, 1, T) – raw EEG
        Returns
        -------
        (B, T, S, 2*F)
        """
        if x.ndim == 3:
            x = x.squeeze(1)            # (B, T)
        B, T = x.shape

        weight = self._build_wavelet_bank(x.device, training)    # (2FS,1,K)

        # (B, 1, T) → (B, 2FS, T)   (depth-wise 아님, 일반 conv)
        conv_out = F.conv1d(
            x.unsqueeze(1),            # (B,1,T)
            weight,
            padding=self.padding
        )

        # (B, 2FS, T) → (B, T, S, 2F)
        F_cnt = weight.shape[0] // (2 * self.scales.numel())
        S = self.scales.numel()
        conv_out = conv_out.view(
            B,                         # batch
            2, F_cnt, S, T             # split real/imag, fb, scale, time
        ).permute(0, 3, 2, 1, 4)        # (B, T, S, 2, F)
        conv_out = conv_out.reshape(B, T, S, -1)  # merge 2, F → 2F

        return conv_out.contiguous()    # (B, T, S, 2F)

class MultiDilatedConvBlock(nn.Module):
    """Multi-dilated convolution block with parallel branches"""
    def __init__(self, in_channels, out_channels, max_dilation, is_2d=False):
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
                    nn.BatchNorm1d(branch_filters, affine=False),
                    nn.ReLU(),
                    nn.Conv1d(branch_filters, branch_filters, kernel_size=3,
                             padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm1d(branch_filters, affine=False),
                    nn.ReLU()
                )
            self.branches.append(branch)
    
    def forward(self, x):
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))
        return torch.cat(outputs, dim=1)


class REDv2Time(nn.Module):
    """REDv2 Time-domain model for K-complex detection"""
    def __init__(self, in_channels=1, params=None):
        super(REDv2Time, self).__init__()
        self.params = params if params is not None else DEFAULT_PARMS
        params = self.params
        
        # Calculate crop sizes
        fs_after_conv = params['fs'] // 8
        self.border_crop_conv = int(np.round(params['border_duration_conv'] * fs_after_conv))
        border_duration_lstm = params['border_duration'] - params['border_duration_conv']
        self.border_crop_lstm = int(np.round(border_duration_lstm * fs_after_conv))
        
        # Input batch normalization
        self.bn_input = nn.BatchNorm1d(in_channels)
        
        # Stem layers
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, params['bigger_stem_filters'], kernel_size=3, padding=1, bias=False),
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
            params['bigger_max_dilation'],
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
            return x[:, :, border_crop:-border_crop]
        return x
    
    def forward(self, x):
        """
        x: (batch_size, 1, time_len)
        Returns: logits (batch_size, time_len//8, 2), probabilities
        """
        # Add channel dimension and apply input batch norm
        # x = x.unsqueeze(1)  # (batch, 1, time)
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
        probabilities = F.softmax(logits, dim=-1)
        
        # return logits, probabilities, {'last_hidden': x}
        return logits


class REDv2CWT1D(nn.Module):
    """REDv2 CWT-domain model for K-complex detection"""
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
        
        # CWT layer
        self.cwt = MorletCWT(
            fb_list         = params['fb_list'],
            fs              = params['fs'],
            lower_freq      = params['lower_freq'],
            upper_freq      = params['upper_freq'],
            n_scales        = params['n_scales'],
            size_factor     = params['wavelet_size_factor'],
            noise_intensity = params['cwt_noise_intensity'],
            trainable       = params['trainable_wavelet']
        )

        # Calculate CWT output channels
        cwt_channels = params['n_scales'] * len(params['fb_list']) * 2  # real + imag
        
        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv1d(cwt_channels, params['bigger_stem_filters'], 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(params['bigger_stem_filters'], affine=False),
            nn.ReLU()
        )
        
        # Multi-dilated convolution blocks
        self.mdconv1 = MultiDilatedConvBlock(
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
        
        self.mdconv2 = MultiDilatedConvBlock(
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
        
        # LSTM layers (same as time model)
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
        
        # Classification layers (same as time model)
        if params['fc_units'] > 0:
            self.fc = nn.Conv1d(lstm_output_size, params['fc_units'], kernel_size=1)
            self.dropout_fc = nn.Dropout(params['drop_rate_hidden'])
            self.fc_activation = nn.ReLU()
            final_size = params['fc_units']
        else:
            self.fc = None
            final_size = lstm_output_size
            
        # Output layer
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
            return x[:, :, border_crop:-border_crop]
        return x
    
    def forward(self, x):
        """
        x: (batch_size, time_len)
        Returns: logits (batch_size, time_len//8, 2), probabilities
        """
        if x.ndim == 3:
            x = x.squeeze(1)

        # Apply CWT
        cwt_output = self.cwt(x, training=self.training)  # (batch, time, scales, channels)
        
        # Flatten scales and channels: (batch, time, scales*channels)
        batch_size, time_len, n_scales, n_channels = cwt_output.shape
        x = cwt_output.view(batch_size, time_len, n_scales * n_channels)
        
        # Transpose for Conv1d: (batch, channels, time)
        x = x.transpose(1, 2)
        
        # Apply border crop from CWT
        border_crop_cwt = int(np.round(self.params['border_duration_cwt'] * self.params['fs']))
        if border_crop_cwt > 0:
            x = x[:, :, border_crop_cwt:-border_crop_cwt]
        
        # Downsample by factor of 2 (stride=2 in original CWT)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        
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
            x = self.dropout1(x)
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
        x = self.dropout_output(x)
        logits = self.output(x)
        
        # Transpose to (batch, time, classes)
        logits = logits.transpose(1, 2)
        probabilities = F.softmax(logits, dim=-1)
        
        return logits


# Example usage
if __name__ == "__main__":
    # Define parameters (from pkeys.default_params)
    params = {
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
    
    # Create models
    model_time = REDv2Time(1, params)
    model_cwt = REDv2CWT1D(params)
    
    # Test with dummy input
    batch_size = 2
    time_len = 5040  # 20 seconds at 200 Hz
    x = torch.randn(batch_size, time_len)

    print("\nTesting REDv2CWT1D model...")
    logits_cwt = model_cwt(x)
    print(f"CWT model output shape: {logits_cwt.shape}")
    
    # Forward pass
    print("Testing REDv2Time model...")
    logits_time, probs_time, outputs_time = model_time(x)
    print(f"Time model output shape: {logits_time.shape}") 
    