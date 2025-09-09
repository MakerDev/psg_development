import torch
import torch.nn as nn
import torch.nn.functional as F

def center_crop_or_pad(x, target_h, target_w):
    B, C, H, W = x.shape

    # --- 세로축(H) ---
    if H > target_h:
        # crop
        diff = H - target_h
        start = diff // 2
        end = start + target_h
        x = x[:, :, start:end, :]
    elif H < target_h:
        # pad
        diff = target_h - H
        pad_before = diff // 2
        pad_after = diff - pad_before
        # (left, right, top, bottom) = (0,0, pad_before, pad_after) for H dimension
        x = F.pad(x, (0, 0, pad_before, pad_after))

    # --- 가로축(W) ---
    if W > target_w:
        # crop
        diff = W - target_w
        start = diff // 2
        end = start + target_w
        x = x[:, :, :, start:end]
    elif W < target_w:
        # pad
        diff = target_w - W
        pad_before = diff // 2
        pad_after = diff - pad_before
        # (left, right, top, bottom) = (pad_before, pad_after, 0, 0) for W dimension
        x = F.pad(x, (pad_before, pad_after, 0, 0))

    return x


class DoubleConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = DoubleConv2D(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)  # (H,W) -> ceil(H/2), ceil(W/2)
        x = self.conv(x)
        return x


class Up2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, 
            in_ch // 2, 
            kernel_size=2, 
            stride=2,
            output_padding=0  # crop/pad로 보정할 것이므로 0
        )
        self.conv = DoubleConv2D(in_ch // 2 + out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)  # (Hup, Wup) = (2*Hd, 2*Wd)
        # center_crop_or_pad로 skip 크기에 맞춤
        # skip.shape[2] = Hs, skip.shape[3] = Ws
        x = center_crop_or_pad(x, skip.shape[2], skip.shape[3])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class DeepSleepSota2D(nn.Module):
    def __init__(self, in_channels=9, out_channels=1, p=0.2):
        super().__init__()
        # ------ Encoder ------
        self.inc   = DoubleConv2D(in_channels, 16)
        self.down1 = Down2D(16, 32)
        self.down2 = Down2D(32, 64)
        self.down3 = Down2D(64, 128)
        self.down4 = Down2D(128, 256)

        # ------ Bottleneck ------
        self.bot   = DoubleConv2D(256, 256)

        # ------ Decoder ------
        self.up4 = Up2D(256, 128)
        self.up3 = Up2D(128, 64)
        self.up2 = Up2D(64, 32)
        self.up1 = Up2D(32, 16)

        self.dropout_enc = nn.Dropout2d(p)
        self.dropout_bot = nn.Dropout2d(p)
        self.dropout_dec = nn.Dropout2d(p)

        # ------ Output ------
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.freq_conv = nn.Conv1d(26, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x, act=True, freq=False):
        x0 = self.inc(x)          # (B,16, H, W)
        x1 = self.down1(x0)       # (B,32,...)
        x1 = self.dropout_enc(x1)

        x2 = self.down2(x1)       # (B,64,...)
        x2 = self.dropout_enc(x2)

        x3 = self.down3(x2)       # (B,128,...)
        x3 = self.dropout_enc(x3)

        x4 = self.down4(x3)       # (B,256,...)
        x4 = self.dropout_enc(x4)

        # Bottleneck
        xb = self.bot(x4)         # (B,256,...)
        xb = self.dropout_bot(xb)

        # Decoder
        xu4 = self.up4(xb, x3)    # (B,128,...)
        xu4 = self.dropout_dec(xu4)

        xu3 = self.up3(xu4, x2)   # (B,64,...)
        xu3 = self.dropout_dec(xu3)

        xu2 = self.up2(xu3, x1)   # (B,32,...)
        xu2 = self.dropout_dec(xu2)

        xu1 = self.up1(xu2, x0)   # (B,16,...)
        xu1 = self.dropout_dec(xu1)

        # ------ Output ------
        out = self.out_conv(xu1)   # (B,1,51,26500)
        if freq:
            out = self.freq_conv(out.squeeze(1))
        if act:
            out = self.act(out)
        return out


if __name__ == "__main__":
    # 간단 테스트
    model = DeepSleepSota2D(in_channels=9, out_channels=1)
    dummy_in = torch.randn(2, 9, 51, 26500)  # 배치=2
    out = model(dummy_in)
    print("Output shape:", out.shape)  
    # -> (2, 1, 51, 26500)
