import torch
import torch.nn as nn
import torch.nn.functional as F

#
# =====================
#     Loss & Metric
# =====================
#

def crossentropy_cut(y_true, y_pred):
    # y_true, y_pred: (N, 1, L)
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)

    # Create the mask
    mask = (y_true_flat >= -0.5)
    # Clip the predictions
    y_pred_clamped = y_pred_flat.clamp(min=1e-7, max=1 - 1e-7)

    # Apply mask
    y_true_masked = y_true_flat[mask]
    y_pred_masked = y_pred_clamped[mask]

    # Binary Cross Entropy
    loss = - (y_true_masked * torch.log(y_pred_masked) +
              (1.0 - y_true_masked) * torch.log(1.0 - y_pred_masked))

    return loss.mean()


def dice_coef(y_true, y_pred, smooth=10.0):
    # Flatten
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)

    # Mask
    mask = (y_true_flat >= -0.5).float()

    # Intersection
    intersection = (y_true_flat * y_pred_flat * mask).sum()

    # Dice
    dice = (2. * intersection + smooth) / (
        (y_true_flat * mask).sum() + (y_pred_flat * mask).sum() + smooth
    )
    return dice


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


#
# =====================
#     Basic Blocks
# =====================
#

class ConvBlock(nn.Module):
    """
    A helper block for 1D convolution + batch-norm + ReLU.
    Emulates: Conv1D(..., kernel_size=7, padding='same') + BatchNorm + ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        # "same" conv in 1D with kernel=7 -> padding = 7//2 = 3
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


#
# =====================
#       U-Net 1D
# =====================
#

class DeepSleepNet1(nn.Module):
    def __init__(self, in_channels=11):
        super().__init__()

        #
        # --- Encoder ---
        #
        # Block 01
        self.conv01_1 = ConvBlock(in_channels, 32, kernel_size=7)
        self.conv01_2 = ConvBlock(32, 32, kernel_size=7)
        self.pool01   = nn.MaxPool1d(kernel_size=2, stride=2)   # L/2

        # Block 0
        self.conv0_1  = ConvBlock(32, 40, kernel_size=7)
        self.conv0_2  = ConvBlock(40, 40, kernel_size=7)
        self.pool0    = nn.MaxPool1d(kernel_size=2, stride=2)   # L/4

        # Block 1
        self.conv1_1  = ConvBlock(40, 48, kernel_size=7)
        self.conv1_2  = ConvBlock(48, 48, kernel_size=7)
        self.pool1    = nn.MaxPool1d(kernel_size=4, stride=4)   # L/16

        # Block 3
        self.conv3_1  = ConvBlock(48, 64, kernel_size=7)
        self.conv3_2  = ConvBlock(64, 64, kernel_size=7)
        self.pool3    = nn.MaxPool1d(kernel_size=4, stride=4)   # L/64

        # Block 5
        self.conv5_1  = ConvBlock(64, 128, kernel_size=7)
        self.conv5_2  = ConvBlock(128, 128, kernel_size=7)
        self.pool5    = nn.MaxPool1d(kernel_size=4, stride=4)   # L/256

        # Block 7
        self.conv7_1  = ConvBlock(128, 256, kernel_size=7)
        self.conv7_2  = ConvBlock(256, 256, kernel_size=7)
        self.pool7    = nn.MaxPool1d(kernel_size=4, stride=4)   # L/1024

        # Block 9
        self.conv9_1  = ConvBlock(256, 500, kernel_size=7)
        self.conv9_2  = ConvBlock(500, 500, kernel_size=7)
        self.pool9    = nn.MaxPool1d(kernel_size=4, stride=4)   # L/4096

        # Block 10 (bottom)
        self.conv10_1 = ConvBlock(500, 1000, kernel_size=7)
        self.conv10_2 = ConvBlock(1000, 1000, kernel_size=7)
        # no pooling here => final L = L/4096

        #
        # --- Decoder ---
        #
        # up11 -> stride=4
        # Output channels for transposed conv is matched to the "left side" channels 
        # we want after concatenation (500).
        self.up11_trans = nn.ConvTranspose1d(
            in_channels=1000, out_channels=500,
            kernel_size=4, stride=4,
            padding=0, output_padding=0
        )
        self.conv11_1   = ConvBlock(500 + 500, 500, kernel_size=7)
        self.conv11_2   = ConvBlock(500, 500, kernel_size=7)

        # up13 -> stride=4
        self.up13_trans = nn.ConvTranspose1d(
            in_channels=500, out_channels=256,
            kernel_size=4, stride=4,
            padding=0, output_padding=0
        )
        self.conv13_1   = ConvBlock(256 + 256, 256, kernel_size=7)
        self.conv13_2   = ConvBlock(256, 256, kernel_size=7)

        # up15 -> stride=4
        self.up15_trans = nn.ConvTranspose1d(
            in_channels=256, out_channels=128,
            kernel_size=4, stride=4,
            padding=0, output_padding=0
        )
        self.conv15_1   = ConvBlock(128 + 128, 128, kernel_size=7)
        self.conv15_2   = ConvBlock(128, 128, kernel_size=7)

        # up17 -> stride=4
        self.up17_trans = nn.ConvTranspose1d(
            in_channels=128, out_channels=64,
            kernel_size=4, stride=4,
            padding=0, output_padding=0
        )
        self.conv17_1   = ConvBlock(64 + 64, 64, kernel_size=7)
        self.conv17_2   = ConvBlock(64, 64, kernel_size=7)

        # up19 -> stride=4
        self.up19_trans = nn.ConvTranspose1d(
            in_channels=64, out_channels=48,
            kernel_size=4, stride=4,
            padding=0, output_padding=0
        )
        self.conv19_1   = ConvBlock(48 + 48, 48, kernel_size=7)
        self.conv19_2   = ConvBlock(48, 48, kernel_size=7)

        # up20 -> stride=2
        self.up20_trans = nn.ConvTranspose1d(
            in_channels=48, out_channels=40,
            kernel_size=2, stride=2,
            padding=0, output_padding=0
        )
        self.conv20_1   = ConvBlock(40 + 40, 40, kernel_size=7)
        self.conv20_2   = ConvBlock(40, 40, kernel_size=7)

        # up21 -> stride=2
        self.up21_trans = nn.ConvTranspose1d(
            in_channels=40, out_channels=32,
            kernel_size=2, stride=2,
            padding=0, output_padding=0
        )
        self.conv21_1   = ConvBlock(32 + 32, 32, kernel_size=7)
        self.conv21_2   = ConvBlock(32, 32, kernel_size=7)

        # final 1D conv
        self.final_conv = nn.Conv1d(32, 1, kernel_size=1)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x):
        """
        x: (N, 11, 2^22)
        Returns: (N, 1, 2^22)
        """
        #
        # --- Encoder ---
        #
        c01 = self.conv01_1(x)     # (N, 32, 2^22)
        c01 = self.conv01_2(c01)   # (N, 32, 2^22)
        p01 = self.pool01(c01)     # (N, 32, 2^21)

        c0 = self.conv0_1(p01)     # (N, 40, 2^21)
        c0 = self.conv0_2(c0)      # (N, 40, 2^21)
        p0 = self.pool0(c0)        # (N, 40, 2^20)

        c1 = self.conv1_1(p0)      # (N, 48, 2^20)
        c1 = self.conv1_2(c1)      # (N, 48, 2^20)
        p1 = self.pool1(c1)        # (N, 48, 2^18)

        c3 = self.conv3_1(p1)      # (N, 64, 2^18)
        c3 = self.conv3_2(c3)      # (N, 64, 2^18)
        p3 = self.pool3(c3)        # (N, 64, 2^16)

        c5 = self.conv5_1(p3)      # (N, 128, 2^16)
        c5 = self.conv5_2(c5)      # (N, 128, 2^16)
        p5 = self.pool5(c5)        # (N, 128, 2^14)

        c7 = self.conv7_1(p5)      # (N, 256, 2^14)
        c7 = self.conv7_2(c7)      # (N, 256, 2^14)
        p7 = self.pool7(c7)        # (N, 256, 2^12)

        c9 = self.conv9_1(p7)      # (N, 500, 2^12)
        c9 = self.conv9_2(c9)      # (N, 500, 2^12)
        p9 = self.pool9(c9)        # (N, 500, 2^10)

        # Bottom
        c10 = self.conv10_1(p9)    # (N, 1000, 2^10)
        c10 = self.conv10_2(c10)   # (N, 1000, 2^10)

        #
        # --- Decoder ---
        #
        # Up to c9 shape
        u11 = self.up11_trans(c10)            # (N, 500, 2^12)
        u11 = torch.cat([u11, c9], dim=1)     # (N, 1000, 2^12)
        c11 = self.conv11_1(u11)              # (N, 500, 2^12)
        c11 = self.conv11_2(c11)              # (N, 500, 2^12)

        # Up to c7 shape
        u13 = self.up13_trans(c11)            # (N, 256, 2^14)
        u13 = torch.cat([u13, c7], dim=1)     # (N, 512, 2^14)
        c13 = self.conv13_1(u13)              # (N, 256, 2^14)
        c13 = self.conv13_2(c13)              # (N, 256, 2^14)

        # Up to c5 shape
        u15 = self.up15_trans(c13)            # (N, 128, 2^16)
        u15 = torch.cat([u15, c5], dim=1)     # (N, 256, 2^16)
        c15 = self.conv15_1(u15)              # (N, 128, 2^16)
        c15 = self.conv15_2(c15)              # (N, 128, 2^16)

        # Up to c3 shape
        u17 = self.up17_trans(c15)            # (N, 64, 2^18)
        u17 = torch.cat([u17, c3], dim=1)     # (N, 128, 2^18)
        c17 = self.conv17_1(u17)              # (N, 64, 2^18)
        c17 = self.conv17_2(c17)              # (N, 64, 2^18)

        # Up to c1 shape
        u19 = self.up19_trans(c17)            # (N, 48, 2^20)
        u19 = torch.cat([u19, c1], dim=1)     # (N, 96, 2^20)
        c19 = self.conv19_1(u19)              # (N, 48, 2^20)
        c19 = self.conv19_2(c19)              # (N, 48, 2^20)

        # Up to c0 shape
        u20 = self.up20_trans(c19)            # (N, 40, 2^21)
        u20 = torch.cat([u20, c0], dim=1)     # (N, 80, 2^21)
        c20 = self.conv20_1(u20)              # (N, 40, 2^21)
        c20 = self.conv20_2(c20)              # (N, 40, 2^21)

        # Up to c01 shape
        u21 = self.up21_trans(c20)            # (N, 32, 2^22)
        u21 = torch.cat([u21, c01], dim=1)    # (N, 64, 2^22)
        c21 = self.conv21_1(u21)              # (N, 32, 2^22)
        c21 = self.conv21_2(c21)              # (N, 32, 2^22)

        # Final 1D conv + sigmoid
        out = self.final_conv(c21)            # (N, 1, 2^22)
        out = self.sigmoid(out)               # (N, 1, 2^22)

        return out


