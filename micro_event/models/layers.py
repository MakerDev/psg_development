import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiDilatedConv1D(nn.Module):
    """Multi-scale dilated convolution block."""
    def __init__(self, in_channels, out_channels, max_dilation=16):
        super().__init__()
        
        # Create branches with different dilations
        self.branches = nn.ModuleList()
        
        num_branches = int(np.log2(max_dilation)) + 1
        for i in range(num_branches):
            dilation = 2 ** i
            branch_channels = out_channels // (2 ** (i + 1))
            if i == num_branches - 1:
                branch_channels = out_channels - sum([out_channels // (2 ** (j + 1)) 
                                                      for j in range(num_branches - 1)])
            
            branch = nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(branch_channels, branch_channels, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
    
    def forward(self, x):
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))
        return torch.cat(outputs, dim=1)

class CWTLayer(nn.Module):
    """Continuous Wavelet Transform layer."""
    def __init__(self, fs, n_scales=32, lower_freq=0.5, upper_freq=30, fb_list=[1.5]):
        super().__init__()
        self.fs = fs
        self.n_scales = n_scales
        self.fb_list = fb_list
        
        # Generate scales
        s_0 = 1 / upper_freq
        s_n = 1 / lower_freq
        base = np.power(s_n / s_0, 1 / (n_scales - 1))
        scales = s_0 * np.power(base, np.arange(n_scales))
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))
        
        # Pre-compute wavelets
        self._create_wavelets()
    
    def _create_wavelets(self):
        """Create Morlet wavelets."""
        wavelets_real = []
        wavelets_imag = []
        
        for fb in self.fb_list:
            max_scale = self.scales[-1]
            kernel_size = int(2 * max_scale * self.fs * np.sqrt(4.5 * fb)) + 1
            
            # Make kernel size odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            t = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size) / self.fs
            
            for scale in self.scales:
                # Morlet wavelet
                norm = 1 / (np.sqrt(np.pi * fb) * scale)
                gauss = torch.exp(-t**2 / (fb * scale**2))
                real = norm * gauss * torch.cos(2 * np.pi * t / scale)
                imag = norm * gauss * torch.sin(2 * np.pi * t / scale)
                
                # Zero-pad to max kernel size
                pad_size = kernel_size - len(real)
                if pad_size > 0:
                    pad_left = pad_size // 2
                    pad_right = pad_size - pad_left
                    real = F.pad(real, (pad_left, pad_right))
                    imag = F.pad(imag, (pad_left, pad_right))
                
                wavelets_real.append(real)
                wavelets_imag.append(imag)
        
        # Stack wavelets: [n_wavelets, kernel_size]
        self.register_buffer('wavelets_real', torch.stack(wavelets_real))
        self.register_buffer('wavelets_imag', torch.stack(wavelets_imag))
    
    def forward(self, x):
        """Apply CWT.
        Args:
            x: Input signal [batch, time]
        Returns:
            cwt: [batch, time, n_scales, 2*n_fb]
        """
        batch_size, time_len = x.shape
        
        # Add channel dimension: [batch, 1, time]
        x = x.unsqueeze(1)
        
        # Prepare wavelets for conv1d: [n_wavelets, 1, kernel_size]
        w_real = self.wavelets_real.unsqueeze(1)
        w_imag = self.wavelets_imag.unsqueeze(1)
        
        # Convolve
        conv_real = F.conv1d(x, w_real, padding=w_real.shape[2]//2, stride=2)
        conv_imag = F.conv1d(x, w_imag, padding=w_imag.shape[2]//2, stride=2)
        
        # Reshape: [batch, n_scales * n_fb, time] -> [batch, time, n_scales, n_fb]
        n_fb = len(self.fb_list)
        conv_real = conv_real.transpose(1, 2).reshape(batch_size, -1, self.n_scales, n_fb)
        conv_imag = conv_imag.transpose(1, 2).reshape(batch_size, -1, self.n_scales, n_fb)
        
        # Concatenate real and imaginary parts
        cwt = torch.cat([conv_real, conv_imag], dim=-1)
        
        return cwt