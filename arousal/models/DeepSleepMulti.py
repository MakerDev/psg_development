import torch
import torch.nn as nn
import torch.nn.functional as F

## ------------------------------------------------------------------
## 1) Time Encoder (1D)
##    - Down blocks => get final (B, c_t, T_t_enc)

class DoubleConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Down1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_ch, out_ch)
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class TimeEncoder(nn.Module):
    """
    (B, in_ch, T) -> (B, out_ch, T//8) for example
    """
    def __init__(self, in_ch=9, base_ch=32):
        super().__init__()
        self.inc = DoubleConv1D(in_ch, base_ch)
        self.down1 = Down1D(base_ch, base_ch*2)
        self.down2 = Down1D(base_ch*2, base_ch*4)
        # bottleneck final
        self.bot = DoubleConv1D(base_ch*4, base_ch*4)

    def forward(self, x):
        # x shape: (B, in_ch, T)
        x0 = self.inc(x)    # (B, base_ch, T)
        x1 = self.down1(x0) # (B, base_ch*2, T/2)
        x2 = self.down2(x1) # (B, base_ch*4, T/4)
        xb = self.bot(x2)   # (B, base_ch*4, T/4)
        return xb   # shape (B, base_ch*4, T/4)


## ------------------------------------------------------------------
## 2) Spectrogram Encoder (2D)
##    - Down blocks => freq/time reduced => freq -> average pool => (B, c_s, time_s_enc)

class DoubleConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Down2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv2D(in_ch, out_ch)
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class SpecEncoder(nn.Module):
    """
    (B, in_ch, freq, time_spec) -> bottleneck(2D) -> freq average pool => (B, out_ch, time_spec//(some factor))
    """
    def __init__(self, in_ch=9, base_ch=32):
        super().__init__()
        self.inc = DoubleConv2D(in_ch, base_ch)
        self.down1 = Down2D(base_ch, base_ch*2)
        self.down2 = Down2D(base_ch*2, base_ch*4)
        self.bot   = DoubleConv2D(base_ch*4, base_ch*4)

    def forward(self, x):
        # x: (B, in_ch, freq, time_spec)
        x0 = self.inc(x)     # (B, base_ch, freq, time_spec)
        x1 = self.down1(x0)  # (B, base_ch*2, freq/2, time_spec/2)
        x2 = self.down2(x1)  # (B, base_ch*4, freq/4, time_spec/4)
        xb = self.bot(x2)    # (B, base_ch*4, freq/4, time_spec/4)

        # freq -> average pool
        # shape => (B, base_ch*4, time_spec/4)
        out_1d = xb.mean(dim=2)  # freq dim=2
        return out_1d


## ------------------------------------------------------------------
## 3) Fuse + Decoder(1D)
##    - time_enc_out: (B, c_t, T_t)
##    - spec_enc_out: (B, c_s, T_s)
##    - match T_t, T_s => crop/pad => concat => decode => (B,1, T_final)

def center_crop_or_pad_1d(x, target_len):
    """
    x: (B, C, L)
    crop or pad in center to match target_len
    """
    B,C,L = x.shape
    if L > target_len:
        diff = L - target_len
        start = diff // 2
        end = start + target_len
        x = x[:,:,start:end]
    elif L < target_len:
        diff = target_len - L
        pad_before = diff//2
        pad_after = diff - pad_before
        x = F.pad(x, (pad_before, pad_after))
    return x


class DoubleConv1D_Decoder(nn.Module):
    """
    간단 Decoder block (실제로는 UpSampling등 하지만,
    여기선 시연을 위해 2-layers conv만)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, in_ch//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_ch//2, out_ch, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return self.conv(x)


class MultiScaleFusionModel(nn.Module):
    """
    - TimeEncoder => (B,c_t, T_t)
    - SpecEncoder => (B,c_s, T_s)
    - crop/pad => concat => 1D decoder => final (B,1,T)
    """
    def __init__(self, in_ch_time=9, in_ch_spec=9, base_ch=32, final_len=2**22):
        super().__init__()
        self.time_enc = TimeEncoder(in_ch=in_ch_time, base_ch=base_ch)
        self.spec_enc = SpecEncoder(in_ch=in_ch_spec, base_ch=base_ch)

        # decoder
        self.decoder = DoubleConv1D_Decoder(in_ch=base_ch*8, out_ch=1)
        # out_ch=1 => arousal mask (B,1,T)

        self.final_len = final_len

    def forward(self, x_time, x_spec):
        """
        x_time: (B, in_ch_time, T)
        x_spec: (B, in_ch_spec, freq, Tspec)
        """
        # 1) encoder
        feat_time = self.time_enc(x_time) # (B, base_ch*4, T_t)
        feat_spec = self.spec_enc(x_spec) # (B, base_ch*4, T_s)

        # 2) match length
        T_t = feat_time.shape[-1]
        T_s = feat_spec.shape[-1]
        # 임의로 T_s -> T_t로 맞추자(혹은 반대)
        if T_t > T_s:
            feat_spec = center_crop_or_pad_1d(feat_spec, T_t)
        elif T_s > T_t:
            feat_time = center_crop_or_pad_1d(feat_time, T_s)

        # 이제 feat_time, feat_spec 둘 중 하나 길이가 같아짐
        # concat in channel dimension
        feat_fused = torch.cat([feat_time, feat_spec], dim=1)  # (B, c_t + c_s, T_?)

        # 3) decoder
        dec_out = self.decoder(feat_fused)  # (B,1, T_?)

        # 4) 최종 출력 길이 -> self.final_len
        out = center_crop_or_pad_1d(dec_out, self.final_len)

        # sigmoid
        out = torch.sigmoid(out)
        return out  # shape (B,1, final_len)

## ------------------------------------------------------------------
## 4) Test
if __name__ == "__main__":
    model = MultiScaleFusionModel(in_ch_time=9, in_ch_spec=9, base_ch=32, final_len=2**22)

    # time input
    x_time = torch.randn(2, 9, 2^22)      # (B=2, channels=9, T=800)
    # spec input
    x_spec = torch.randn(2, 9, 16, 355000)  # (B=2, channels=9, freq=32, time_spec=200)

    out = model(x_time, x_spec)
    print("Output shape:", out.shape)  # (2,1,512)
