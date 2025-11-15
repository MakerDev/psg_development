"""
DeepSleepFinal: Multimodal Arousal Detection Model

This model combines time-domain and frequency-domain information:
1. Time Branch: Processes raw signal + amplitude features using 1D U-Net
2. Frequency Branch: Processes spectrogram using 2D U-Net
3. Fusion Module: Combines features using cross-attention mechanism

The dual-branch architecture captures both:
- Abrupt amplitude changes (time domain)
- Frequency shifts (frequency domain)

This multimodal approach provides more robust arousal detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== 1D TIME BRANCH ====================

def center_crop_or_pad_1d(x, target_t):
    """Center crop or pad 1D tensor to target length"""
    B, C, T = x.shape
    if T > target_t:
        diff = T - target_t
        start = diff // 2
        end = start + target_t
        x = x[:, :, start:end]
    elif T < target_t:
        diff = target_t - T
        pad_before = diff // 2
        pad_after = diff - pad_before
        x = F.pad(x, (pad_before, pad_after))
    return x


class DoubleConv1D(nn.Module):
    """Two consecutive 1D convolutions with BatchNorm and ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=21, padding=10, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down1D(nn.Module):
    """Downsampling block: MaxPool + DoubleConv"""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_ch, out_ch, dropout=dropout)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up1D(nn.Module):
    """Upsampling block: ConvTranspose + Skip Connection + DoubleConv"""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_ch//2 + out_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = center_crop_or_pad_1d(x, skip.shape[2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class TimeBranchUNet(nn.Module):
    """
    1D U-Net for time-domain features
    Input: (B, C*6, T) where C=9 channels, 6 features per channel
    Output: (B, 1, T) arousal prediction
    """
    def __init__(self, n_channels=9, n_features=6, base_ch=16, dropout=0.15):
        super().__init__()
        in_ch = n_channels * n_features  # 9 * 6 = 54

        # Encoder
        self.inc = DoubleConv1D(in_ch, base_ch, dropout=dropout)
        self.down1 = Down1D(base_ch, base_ch*2, dropout=dropout)
        self.down2 = Down1D(base_ch*2, base_ch*4, dropout=dropout)
        self.down3 = Down1D(base_ch*4, base_ch*8, dropout=dropout)
        self.down4 = Down1D(base_ch*8, base_ch*16, dropout=dropout)
        self.down5 = Down1D(base_ch*16, base_ch*32, dropout=dropout)

        # Bottleneck
        self.bot = DoubleConv1D(base_ch*32, base_ch*32, dropout=dropout)

        # Decoder
        self.up1 = Up1D(base_ch*32, base_ch*16, dropout=dropout)
        self.up2 = Up1D(base_ch*16, base_ch*8, dropout=dropout)
        self.up3 = Up1D(base_ch*8, base_ch*4, dropout=dropout)
        self.up4 = Up1D(base_ch*4, base_ch*2, dropout=dropout)
        self.up5 = Up1D(base_ch*2, base_ch, dropout=dropout)

        # Output
        self.out_conv = nn.Conv1d(base_ch, base_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        # Bottleneck
        xb = self.bot(x5)

        # Decoder with skip connections
        xu1 = self.up1(xb, x4)
        xu2 = self.up2(xu1, x3)
        xu3 = self.up3(xu2, x2)
        xu4 = self.up4(xu3, x1)
        xu5 = self.up5(xu4, x0)

        out = self.out_conv(xu5)
        return out


# ==================== 2D FREQUENCY BRANCH ====================

def center_crop_or_pad_2d(x, target_h, target_w):
    """Center crop or pad 2D tensor to target shape"""
    B, C, H, W = x.shape

    # Height
    if H > target_h:
        diff = H - target_h
        start = diff // 2
        x = x[:, :, start:start+target_h, :]
    elif H < target_h:
        diff = target_h - H
        pad_before = diff // 2
        pad_after = diff - pad_before
        x = F.pad(x, (0, 0, pad_before, pad_after))

    # Width
    if W > target_w:
        diff = W - target_w
        start = diff // 2
        x = x[:, :, :, start:start+target_w]
    elif W < target_w:
        diff = target_w - W
        pad_before = diff // 2
        pad_after = diff - pad_before
        x = F.pad(x, (pad_before, pad_after, 0, 0))

    return x


class DoubleConv2D(nn.Module):
    """Two consecutive 2D convolutions with BatchNorm and ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down2D(nn.Module):
    """Downsampling block: MaxPool + DoubleConv"""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = DoubleConv2D(in_ch, out_ch, dropout=dropout)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up2D(nn.Module):
    """Upsampling block: ConvTranspose + Skip Connection + DoubleConv"""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv2D(in_ch//2 + out_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = center_crop_or_pad_2d(x, skip.shape[2], skip.shape[3])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class FrequencyBranchUNet(nn.Module):
    """
    2D U-Net for frequency-domain features (spectrogram)
    Input: (B, C, F, T) where C=9 channels, F=freq bins, T=time bins
    Output: (B, out_ch, F, T) feature map
    """
    def __init__(self, n_channels=9, base_ch=16, num_layers=4, dropout=0.15):
        super().__init__()

        # Encoder
        self.inc = DoubleConv2D(n_channels, base_ch, dropout=dropout)
        self.downs = nn.ModuleList()
        ch = base_ch
        for _ in range(num_layers):
            self.downs.append(Down2D(ch, ch*2, dropout=dropout))
            ch *= 2

        # Bottleneck
        self.bot = DoubleConv2D(ch, ch, dropout=dropout)

        # Decoder
        self.ups = nn.ModuleList()
        for i in range(num_layers-1, -1, -1):
            prev_ch = ch
            skip_ch = base_ch * (2**i)
            self.ups.append(Up2D(prev_ch, skip_ch, dropout=dropout))
            ch = skip_ch

        self.out_conv = nn.Conv2d(base_ch, base_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Bottleneck
        x = self.bot(x)

        # Decoder
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)

        out = self.out_conv(x)
        return out


# ==================== CROSS-ATTENTION FUSION ====================

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module to fuse time and frequency features

    Allows time features to attend to frequency features and vice versa
    """
    def __init__(self, time_ch=16, freq_ch=16, hidden_dim=32):
        super().__init__()

        # Project features to common dimension
        self.time_proj = nn.Conv1d(time_ch, hidden_dim, kernel_size=1)
        self.freq_proj = nn.Conv1d(freq_ch, hidden_dim, kernel_size=1)

        # Attention weights
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, time_feat, freq_feat):
        """
        Args:
            time_feat: (B, C_time, T)
            freq_feat: (B, C_freq, F, T_spec)
        Returns:
            fused: (B, 1, T)
        """
        B, _, T = time_feat.shape

        # Average pool frequency dimension of spectrogram features
        freq_feat = freq_feat.mean(dim=2)  # (B, C_freq, T_spec)

        # Interpolate freq_feat to match time_feat length
        if freq_feat.shape[2] != T:
            freq_feat = F.interpolate(freq_feat, size=T, mode='linear', align_corners=False)

        # Project to common dimension
        time_proj = self.time_proj(time_feat)  # (B, hidden, T)
        freq_proj = self.freq_proj(freq_feat)  # (B, hidden, T)

        # Reshape for attention: (B, T, hidden)
        time_seq = time_proj.permute(0, 2, 1)
        freq_seq = freq_proj.permute(0, 2, 1)

        # Cross-attention: time attends to freq
        time_attn, _ = self.attn(time_seq, freq_seq, freq_seq)

        # Cross-attention: freq attends to time
        freq_attn, _ = self.attn(freq_seq, time_seq, time_seq)

        # Concatenate attended features
        fused_seq = torch.cat([time_attn, freq_attn], dim=2)  # (B, T, hidden*2)
        fused = fused_seq.permute(0, 2, 1)  # (B, hidden*2, T)

        # Output projection
        out = self.out_proj(fused)  # (B, 1, T)

        return out


# ==================== MAIN MULTIMODAL MODEL ====================

class DeepSleepFinal(nn.Module):
    """
    Multimodal Arousal Detection Model

    Combines time-domain and frequency-domain processing with cross-attention fusion

    Args:
        n_channels: number of EEG channels (default: 9)
        n_time_features: number of time-domain features per channel (default: 6)
        time_base_ch: base channels for time branch (default: 16)
        freq_base_ch: base channels for frequency branch (default: 16)
        freq_layers: number of layers in frequency branch (default: 4)
        dropout: dropout rate (default: 0.15)
    """
    def __init__(self,
                 n_channels=9,
                 n_time_features=6,
                 time_base_ch=16,
                 freq_base_ch=16,
                 freq_layers=4,
                 dropout=0.15,
                 chunk_size=2**17,
                 overlap=0.25):
        super().__init__()

        self.chunk_size = chunk_size  # Process in chunks to save memory
        self.overlap = overlap  # Overlap ratio between chunks

        # Time branch (1D U-Net)
        self.time_branch = TimeBranchUNet(
            n_channels=n_channels,
            n_features=n_time_features,
            base_ch=time_base_ch,
            dropout=dropout
        )

        # Frequency branch (2D U-Net)
        self.freq_branch = FrequencyBranchUNet(
            n_channels=n_channels,
            base_ch=freq_base_ch,
            num_layers=freq_layers,
            dropout=dropout
        )

        # Fusion module
        self.fusion = CrossAttentionFusion(
            time_ch=time_base_ch,
            freq_ch=freq_base_ch,
            hidden_dim=32
        )

        # Final activation
        self.sigmoid = nn.Sigmoid()

    def _process_chunk(self, x_time_chunk, x_freq_chunk, apply_sigmoid):
        """Process a single chunk"""
        # Process time features
        time_feat = self.time_branch(x_time_chunk)

        # Process frequency features
        freq_feat = self.freq_branch(x_freq_chunk)

        # Fuse features
        out = self.fusion(time_feat, freq_feat)

        # Apply activation
        if apply_sigmoid:
            out = self.sigmoid(out)

        return out

    def forward(self, x_time, x_freq, apply_sigmoid=True, use_chunking=True):
        """
        Args:
            x_time: (B, C*6, T) time-domain features
            x_freq: (B, C, F, T_spec) frequency-domain features
            apply_sigmoid: whether to apply sigmoid activation
            use_chunking: whether to use chunking (default True for memory efficiency)

        Returns:
            out: (B, 1, T) arousal predictions
        """
        B, C_time, T = x_time.shape
        _, C_freq, F, T_spec = x_freq.shape

        # If input is small enough or chunking disabled, process directly
        if not use_chunking or T <= self.chunk_size:
            return self._process_chunk(x_time, x_freq, apply_sigmoid)

        # Otherwise, process in overlapping chunks
        hop_size = int(self.chunk_size * (1 - self.overlap))
        spec_ratio = T_spec / T  # Spectrogram to time ratio

        # Output tensor
        outputs = []
        weights = []

        # Process chunks with overlap
        start = 0
        while start < T:
            end = min(start + self.chunk_size, T)

            # Extract time chunk
            x_time_chunk = x_time[:, :, start:end]

            # Extract corresponding spectrogram chunk
            spec_start = int(start * spec_ratio)
            spec_end = int(end * spec_ratio)
            x_freq_chunk = x_freq[:, :, :, spec_start:spec_end]

            # Process chunk
            with torch.cuda.amp.autocast(enabled=False):  # Disable AMP for stability
                chunk_out = self._process_chunk(x_time_chunk, x_freq_chunk, apply_sigmoid)

            # Pad chunk output to match chunk size if needed
            chunk_len = chunk_out.shape[2]
            if chunk_len < (end - start):
                pad_len = (end - start) - chunk_len
                chunk_out = F.pad(chunk_out, (0, pad_len), mode='constant', value=0)
            elif chunk_len > (end - start):
                chunk_out = chunk_out[:, :, :(end - start)]

            outputs.append(chunk_out)

            # Create weight for blending (linear fade in/out at boundaries)
            weight = torch.ones(1, 1, chunk_out.shape[2], device=x_time.device)
            fade_len = int(hop_size * self.overlap)

            if start > 0:  # Fade in at start
                fade_in = torch.linspace(0, 1, fade_len, device=x_time.device)
                weight[:, :, :fade_len] = fade_in

            if end < T:  # Fade out at end
                fade_out = torch.linspace(1, 0, fade_len, device=x_time.device)
                weight[:, :, -fade_len:] = fade_out

            weights.append(weight)

            # Move to next chunk
            if end >= T:
                break
            start += hop_size

        # Reconstruct full output by blending overlapping chunks
        output_full = torch.zeros(B, 1, T, device=x_time.device)
        weight_full = torch.zeros(B, 1, T, device=x_time.device)

        start = 0
        for chunk_out, weight in zip(outputs, weights):
            chunk_len = chunk_out.shape[2]
            end = min(start + chunk_len, T)
            actual_len = end - start

            output_full[:, :, start:end] += chunk_out[:, :, :actual_len] * weight[:, :, :actual_len]
            weight_full[:, :, start:end] += weight[:, :, :actual_len]

            start += hop_size

        # Normalize by weight (avoid division by zero)
        weight_full = torch.clamp(weight_full, min=1e-8)
        output_full = output_full / weight_full

        return output_full


if __name__ == "__main__":
    # Test the model
    batch_size = 2
    n_channels = 9
    n_time_features = 6
    time_length = 2**21  # ~40 seconds at 50 Hz
    freq_bins = 51
    spec_time_bins = time_length // 50  # depends on spectrogram parameters

    # Create dummy inputs
    x_time = torch.randn(batch_size, n_channels * n_time_features, time_length)
    x_freq = torch.randn(batch_size, n_channels, freq_bins, spec_time_bins)

    # Create model
    model = DeepSleepFinal(
        n_channels=9,
        n_time_features=6,
        time_base_ch=16,
        freq_base_ch=16,
        freq_layers=4,
        dropout=0.15
    )

    # Forward pass
    output = model(x_time, x_freq)

    print(f"Input time shape: {x_time.shape}")
    print(f"Input freq shape: {x_freq.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
