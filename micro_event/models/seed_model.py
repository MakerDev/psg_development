import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvMDBlock(nn.Module):
    """Convolutional Multi-Dilated Block: four parallel 2-layer conv branches with different dilation rates."""
    def __init__(self, in_channels, out_channels):
        super(ConvMDBlock, self).__init__()
        # We will split out_channels into 4 branches
        assert out_channels % 4 == 0, "out_channels should be divisible by 4 for 4 parallel branches"
        branch_out = out_channels // 4
        # Dilation rates for the four branches (as in Fig.2c of the paper)
        dilations = [1, 2, 4, 8]
        self.branches = nn.ModuleList()
        for d in dilations:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, branch_out, kernel_size=3, dilation=d, padding=d),  # padding=d to keep length
                nn.BatchNorm1d(branch_out),
                nn.ReLU(inplace=True),
                nn.Conv1d(branch_out, branch_out, kernel_size=3, dilation=d, padding=d),   # second conv in branch
                nn.BatchNorm1d(branch_out),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        # After parallel branches, we will combine their outputs (by summation or concatenation).
        # Here we choose to concatenate and then use a 1x1 conv to fuse if needed.
        self.fuse = None
        if True:  # Option: concatenate then fuse to out_channels (makes parameters count similar to single branch)
            self.fuse = nn.Conv1d(out_channels, out_channels, kernel_size=1)  # fuse 4*branch_out -> out_channels

    def forward(self, x):
        # x shape: (batch, in_channels, seq_len)
        branch_outs = [branch(x) for branch in self.branches]  # list of (batch, branch_out, seq_len)
        # Concatenate along channel dimension
        x_cat = torch.cat(branch_outs, dim=1)  # shape: (batch, 4*branch_out == out_channels, seq_len)
        if self.fuse:
            x_cat = self.fuse(x_cat)
        return x_cat

class SEEDModel(nn.Module):
    """Sleep EEG Event Detector (SEED) model: CNN + BLSTM + classifier for spindles and K-complexes."""
    def __init__(self, in_channels=6, time_downsample=8):
        """
        :param in_channels: number of EEG channels input (default 6).
        :param time_downsample: overall downsampling factor in time (default 8 as per SEED design).
        """
        super(SEEDModel, self).__init__()
        # Input batchnorm (normalize each channel's distribution)
        self.input_bn = nn.BatchNorm1d(in_channels)
        # Local Encoding CNN Stage
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)   # first conv layer
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)            # second conv layer (keeping 64 channels)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2_bn = nn.BatchNorm1d(64)
        # Two multi-dilation conv blocks
        self.conv_mdb1 = ConvMDBlock(64, 128)   # outputs 128 channels
        self.conv_mdb2 = ConvMDBlock(128, 256)  # outputs 256 channels

        # BLSTM Contextualization Stage
        # Use batch_first=True for convenience (input shape: batch, seq, features)
        self.blstm1 = nn.LSTM(input_size=256, hidden_size=256, bidirectional=True, batch_first=True)
        self.blstm2 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        # Dropouts for BLSTM outputs
        self.dropout_blstm1 = nn.Dropout(p=0.1)
        self.dropout_blstm2 = nn.Dropout(p=0.1)
        # 1x1 conv to reduce features to 128 (from 512 BLSTM output)
        self.lin_proj = nn.Conv1d(512, 128, kernel_size=1)  # acts as linear layer per time step
        self.lin_proj_bn = nn.BatchNorm1d(128)
        self.relu_proj = nn.ReLU(inplace=True)
        self.dropout_proj = nn.Dropout(p=0.1)

        # Classification stage: 1x1 conv to 2 outputs (spindle vs background, KC vs background)
        # Using two outputs with Sigmoid for multi-label classification
        self.classifier = nn.Conv1d(128, 2, kernel_size=1)
        # (Sigmoid will be applied in forward)
    
    def forward(self, x):
        """
        Forward pass of SEED model.
        :param x: Tensor of shape (batch, 6, N) where N is the number of time samples (e.g. 5040 for 20s+context).
        :return: Tensor of shape (batch, 2, N/8) with sigmoid probabilities for spindle and K-complex at each output time step.
        """
        # Input batch normalization (channel-wise)
        x = self.input_bn(x)
        # Initial conv layers
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        # First pooling (downsample by 2)
        x = F.avg_pool1d(x, kernel_size=2)
        # First Conv-MDB block + pool
        x = self.conv_mdb1(x)
        x = F.avg_pool1d(x, kernel_size=2)
        # Second Conv-MDB block + pool
        x = self.conv_mdb2(x)
        x = F.avg_pool1d(x, kernel_size=2)
        # Now x shape: (batch, 256, seq_len/8). For 20s window, seq_len≈4000, so output length ~500.
        # Transpose for LSTM: (batch, seq, features)
        x = x.transpose(1, 2)  # shape (batch, seq_len_down, 256)
        # BLSTM layers
        x, _ = self.blstm1(x)    # output shape: (batch, seq_len_down, 512)
        x = self.dropout_blstm1(x)
        x, _ = self.blstm2(x)    # output shape: (batch, seq_len_down, 512)
        x = self.dropout_blstm2(x)
        # Project features down to 128 via 1x1 conv (operate on (batch, features, seq) so transpose back)
        x = x.transpose(1, 2)  # (batch, 512, seq_len_down)
        x = self.relu_proj(self.lin_proj_bn(self.lin_proj(x)))  # (batch, 128, seq_len_down)
        x = self.dropout_proj(x)
        # Final classifier conv to 2 outputs
        logits = self.classifier(x)   # shape: (batch, 2, seq_len_down)
        # Apply Sigmoid to get probabilities in [0,1]
        # prob = torch.sigmoid(logits)
        return logits.transpose(1, 2)

if __name__ == "__main__":
    # Example usage
    model = SEEDModel(in_channels=1)
    input_tensor = torch.randn(8, 1, 5040)  # batch size 8, 1 channel, 5040 samples (20s at 252Hz)
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Should be (8, 2, 630) for downsampled length