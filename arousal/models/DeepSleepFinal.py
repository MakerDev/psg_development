import torch
import torch.nn as nn
import torch.nn.functional as F


def center_crop_or_pad_1d(x, target_t):
    """
    Center crop or pad 1D tensor
    x: Tensor of shape (B, C, T)
    target_t: Target sequence length
    """
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


# ==================== Time Domain Branch ====================
class TimeDomainBranch(nn.Module):
    """
    Process time-domain features: raw signal + envelope + derivatives
    Input: (B, C, 4, T) where 4 = [raw, envelope, 1st_deriv, 2nd_deriv]
    """
    def __init__(self, n_channels=9, base_ch=32):
        super().__init__()

        # Process each feature type separately then combine
        self.feature_conv = nn.Sequential(
            nn.Conv2d(n_channels, base_ch, kernel_size=(4, 1), stride=1, padding=0),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )  # (B, base_ch, 1, T)

        # 1D convolution for temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch * 2, kernel_size=21, padding=10),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_ch * 2, base_ch * 2, kernel_size=21, padding=10),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
        )

        # Downsample
        self.down1 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=15, padding=7),
            nn.BatchNorm1d(base_ch * 4),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(base_ch * 4, base_ch * 8, kernel_size=11, padding=5),
            nn.BatchNorm1d(base_ch * 8),
            nn.ReLU(inplace=True)
        )

        self.out_channels = base_ch * 8

    def forward(self, x):
        # x: (B, C, 4, T)
        B, C, F, T = x.shape

        # Process features
        x = self.feature_conv(x)  # (B, base_ch, 1, T)
        x = x.squeeze(2)  # (B, base_ch, T)

        # Temporal processing
        x = self.temporal_conv(x)  # (B, base_ch*2, T)
        x = self.down1(x)  # (B, base_ch*4, T//2)
        x = self.down2(x)  # (B, base_ch*8, T//4)

        return x


# ==================== Frequency Domain Branch ====================
class FrequencyDomainBranch(nn.Module):
    """
    Process spectrogram features
    Input: (B, C, F, T) where F=freq_bins, T=time_bins
    """
    def __init__(self, n_channels=9, base_ch=32):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True)
        )

        # Global pooling along frequency axis
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        self.out_channels = base_ch * 8

    def forward(self, x):
        # x: (B, C, F, T)
        x = self.conv1(x)   # (B, base_ch, F, T)
        x = self.down1(x)   # (B, base_ch*2, F//2, T//2)
        x = self.down2(x)   # (B, base_ch*4, F//4, T//4)
        x = self.down3(x)   # (B, base_ch*8, F//8, T//8)

        # Pool frequency dimension
        x = self.freq_pool(x)  # (B, base_ch*8, 1, T//8)
        x = x.squeeze(2)       # (B, base_ch*8, T//8)

        return x


# ==================== Amplitude Features Branch ====================
class AmplitudeBranch(nn.Module):
    """
    Process statistical/amplitude features
    Input: (B, C, n_features, T_windows)
    """
    def __init__(self, n_channels=9, n_features=6, base_ch=32):
        super().__init__()

        # Combine channel and feature dimensions
        self.feature_conv = nn.Sequential(
            nn.Conv2d(n_channels, base_ch, kernel_size=(n_features, 1), padding=0),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )  # (B, base_ch, 1, T_windows)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_ch * 4),
            nn.ReLU(inplace=True)
        )

        self.out_channels = base_ch * 4

    def forward(self, x):
        # x: (B, C, n_features, T_windows)
        x = self.feature_conv(x)  # (B, base_ch, 1, T_windows)
        x = x.squeeze(2)          # (B, base_ch, T_windows)
        x = self.temporal_conv(x) # (B, base_ch*4, T_windows)

        return x


# ==================== Attention Fusion Module ====================
class AttentionFusion(nn.Module):
    """
    Attention-based fusion of multiple modalities
    """
    def __init__(self, time_ch, freq_ch, amp_ch):
        super().__init__()

        total_ch = time_ch + freq_ch + amp_ch

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(total_ch, total_ch // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(total_ch // 4, total_ch, kernel_size=1),
            nn.Sigmoid()
        )

        # Modality-specific gates
        self.time_gate = nn.Parameter(torch.ones(1))
        self.freq_gate = nn.Parameter(torch.ones(1))
        self.amp_gate = nn.Parameter(torch.ones(1))

    def forward(self, time_feat, freq_feat, amp_feat):
        """
        time_feat: (B, C_time, T)
        freq_feat: (B, C_freq, T)
        amp_feat: (B, C_amp, T)
        """
        # Apply learnable gates
        time_feat = time_feat * torch.sigmoid(self.time_gate)
        freq_feat = freq_feat * torch.sigmoid(self.freq_gate)
        amp_feat = amp_feat * torch.sigmoid(self.amp_gate)

        # Concatenate
        fused = torch.cat([time_feat, freq_feat, amp_feat], dim=1)  # (B, C_total, T)

        # Channel attention
        attention = self.channel_attention(fused)  # (B, C_total, 1)
        fused = fused * attention  # (B, C_total, T)

        return fused


# ==================== Main Multimodal Model ====================
class DeepSleepMultimodal(nn.Module):
    """
    Multimodal arousal detection model combining:
    - Time domain features (raw + envelope + derivatives)
    - Frequency domain features (spectrogram)
    - Amplitude/statistical features
    """
    def __init__(self, n_channels=9, base_ch=32, use_attention=True):
        super().__init__()

        self.use_attention = use_attention

        # Three branches
        self.time_branch = TimeDomainBranch(n_channels, base_ch)
        self.freq_branch = FrequencyDomainBranch(n_channels, base_ch)
        self.amp_branch = AmplitudeBranch(n_channels, n_features=6, base_ch=base_ch)

        # Fusion
        if use_attention:
            self.fusion = AttentionFusion(
                self.time_branch.out_channels,
                self.freq_branch.out_channels,
                self.amp_branch.out_channels
            )

        total_ch = (self.time_branch.out_channels +
                   self.freq_branch.out_channels +
                   self.amp_branch.out_channels)

        # Final processing
        self.final_conv = nn.Sequential(
            nn.Conv1d(total_ch, 256, kernel_size=11, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # Upsampling to match original time resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Output
        self.out_conv = nn.Conv1d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def align_features(self, time_feat, freq_feat, amp_feat):
        """Align all features to same temporal dimension"""
        # Find minimum temporal dimension
        min_t = min(time_feat.shape[2], freq_feat.shape[2], amp_feat.shape[2])

        # Resize all to minimum
        if time_feat.shape[2] != min_t:
            time_feat = F.interpolate(time_feat, size=min_t, mode='linear', align_corners=False)
        if freq_feat.shape[2] != min_t:
            freq_feat = F.interpolate(freq_feat, size=min_t, mode='linear', align_corners=False)
        if amp_feat.shape[2] != min_t:
            amp_feat = F.interpolate(amp_feat, size=min_t, mode='linear', align_corners=False)

        return time_feat, freq_feat, amp_feat

    def forward(self, x_time_combined, x_spec, x_stat, comp=True):
        """
        Args:
            x_time_combined: (B, C, 4, T) - time domain features
            x_spec: (B, C, F, T_spec) - spectrogram
            x_stat: (B, C, n_features, T_windows) - statistical features
            comp: whether to apply sigmoid (for compatibility)

        Returns:
            (B, 1, T) - arousal predictions
        """
        # Process each branch
        time_feat = self.time_branch(x_time_combined)  # (B, C_time, T//4)
        freq_feat = self.freq_branch(x_spec)           # (B, C_freq, T_spec//8)
        amp_feat = self.amp_branch(x_stat)             # (B, C_amp, T_windows)

        # Align temporal dimensions
        time_feat, freq_feat, amp_feat = self.align_features(time_feat, freq_feat, amp_feat)

        # Fusion
        if self.use_attention:
            fused = self.fusion(time_feat, freq_feat, amp_feat)
        else:
            fused = torch.cat([time_feat, freq_feat, amp_feat], dim=1)

        # Final processing
        x = self.final_conv(fused)  # (B, 64, T')
        x = self.upsample(x)        # (B, 16, T'*4)

        # Output
        out = self.out_conv(x)      # (B, 1, T_out)

        if comp:
            out = self.sigmoid(out)

        return out


# ==================== Testing ====================
if __name__ == "__main__":
    # Test the model
    batch_size = 2
    n_channels = 9
    time_samples = 3000  # 60 seconds at 50Hz

    # Create dummy inputs
    x_time_combined = torch.randn(batch_size, n_channels, 4, time_samples)
    x_spec = torch.randn(batch_size, n_channels, 51, 119)  # freq=51, time=119 for 60s
    x_stat = torch.randn(batch_size, n_channels, 6, 59)     # 6 features, 59 windows

    # Create model
    model = DeepSleepMultimodal(n_channels=9, base_ch=32, use_attention=True)

    # Forward pass
    output = model(x_time_combined, x_spec, x_stat, comp=True)

    print(f"Input shapes:")
    print(f"  Time combined: {x_time_combined.shape}")
    print(f"  Spectrogram: {x_spec.shape}")
    print(f"  Statistical: {x_stat.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
