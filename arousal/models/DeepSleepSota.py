import torch
import torch.nn as nn
import torch.nn.functional as F




def center_crop_or_pad_1d(x, target_t):
    """
    x: Tensor of shape (B, C, T)
    target_t: 원하는 시퀀스 길이
    """
    B, C, T = x.shape

    if T > target_t:
        # 중앙에서 target_t 길이만큼 자르기
        diff = T - target_t
        start = diff // 2
        end = start + target_t
        x = x[:, :, start:end]

    elif T < target_t:
        # 중앙에 맞춰 pad
        diff = target_t - T
        pad_before = diff // 2
        pad_after = diff - pad_before
        x = F.pad(x, (pad_before, pad_after))

    return x


class DoubleConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=21, padding=10):
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


class Up1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_ch//2 + out_ch, out_ch)  
    
    def forward(self, x, skip):
        x = self.up(x)
        x = center_crop_or_pad_1d(x, skip.shape[2])
        x = torch.cat([skip, x], dim=1) 
        x = self.conv(x)
        return x


class DeepSleepNetSota(nn.Module):
    def __init__(self, n_channels=9):
        super().__init__()

        self.inc    = DoubleConv1D(n_channels,   20) 
        self.down1  = Down1D(20,  25)
        self.down2  = Down1D(25,  30)
        self.down3  = Down1D(30,  60)
        self.down4  = Down1D(60,  120)
        self.down5  = Down1D(120, 240)
        self.down6  = Down1D(240, 480)

        self.bot    = DoubleConv1D(480, 480)

        self.up1  = Up1D(480, 240)
        self.up2  = Up1D(240, 120)
        self.up3  = Up1D(120, 60)
        self.up4  = Up1D(60,  30)
        self.up5  = Up1D(30,  25)
        self.up6  = Up1D(25,  20)

        self.out_conv = nn.Conv1d(20, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x, comp=True):
        # down path
        x0 = self.inc(x)    
        x1 = self.down1(x0) 
        x2 = self.down2(x1) 
        x3 = self.down3(x2) 
        x4 = self.down4(x3) 
        x5 = self.down5(x4) 
        x6 = self.down6(x5) 

        # bottleneck
        xb = self.bot(x6)   

        xu1 = self.up1(xb, x5) 
        xu2 = self.up2(xu1, x4)
        xu3 = self.up3(xu2, x3)
        xu4 = self.up4(xu3, x2)
        xu5 = self.up5(xu4, x1)
        xu6 = self.up6(xu5, x0)

        out = self.out_conv(xu6)
        out = self.act(out)
        return out

if __name__ == "__main__":
    # Example usage
    model = DeepSleepNetSota(n_channels=9)
    x = torch.randn(1, 9, 2**21) 
    output = model(x) # 
    print(output.shape) 