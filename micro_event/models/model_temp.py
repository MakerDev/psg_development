# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SleepEventDetector(nn.Module):
    def __init__(self, input_channels=1, lstm_hidden=256):
        super(SleepEventDetector, self).__init__()
        # Convolutional encoder: downsample time by factor 8
        # Conv layers to extract local features
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, stride=2, padding=2),  # 1 -> 16 channels
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # 16 -> 32
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # 32 -> 64
            nn.ReLU(),
            # Dilated conv layers (no further downsampling) to increase receptive field ~1s
            nn.Conv1d(64, 64, kernel_size=3, dilation=2, padding=2),  # dilation 2
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, dilation=4, padding=4),  # dilation 4
            nn.ReLU(),
            # (Further dilation layers could be added if needed for larger context)
        )
        
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=lstm_hidden, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=2*lstm_hidden, hidden_size=lstm_hidden, bidirectional=True, batch_first=True)
        # 1x1 convolution to reduce features to 128 channels:contentReference[oaicite:27]{index=27}
        self.conv_reduce = nn.Conv1d(2*lstm_hidden, 128, kernel_size=1)
        # Final 1x1 convolution to output 2 class scores (background vs event)
        self.conv_out = nn.Conv1d(128, 2, kernel_size=1)
        # Dropout probabilities for LSTM outputs and after conv_reduce
        self.dropout_lstm1 = 0.2
        self.dropout_lstm2 = 0.5
        self.dropout_conv = 0.5

    def _init_parameters(self):
        """
        Initialize model parameters.
        This can be customized if needed, e.g., using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        """
        Forward pass.
        x shape: (batch, 1, N) where N is the number of samples in the 20s window (e.g., 5120 at 256 Hz).
        Returns: (batch, T_out, 2) class score logits for each time step (T_out ~ N/8).
        """
        # Apply convolutional encoder
        # Input x: [B, 1, N]; output conv_x: [B, 64, N/8] (after three stride-2 convs)
        conv_x = self.conv_net(x)
        # Transpose to [B, T_out, features] for LSTM (T_out = conv output length)
        conv_x = conv_x.transpose(1, 2)  # shape [B, T_out, 64]
        # BiLSTM layers
        lstm_out1, _ = self.lstm1(conv_x)            # [B, T_out, 2*lstm_hidden]
        lstm_out1 = F.dropout(lstm_out1, p=self.dropout_lstm1, training=self.training)
        lstm_out2, _ = self.lstm2(lstm_out1)         # [B, T_out, 2*lstm_hidden]
        lstm_out2 = F.dropout(lstm_out2, p=self.dropout_lstm2, training=self.training)
        # Conv reduce and output layers
        # Transpose LSTM output to [B, features, T_out] for conv layers
        lstm_out2_t = lstm_out2.transpose(1, 2)      # [B, 2*lstm_hidden, T_out]
        features = F.relu(self.conv_reduce(lstm_out2_t))  # [B, 128, T_out]
        features = F.dropout(features, p=self.dropout_conv, training=self.training)
        scores = self.conv_out(features)             # [B, 2, T_out] (class scores)
        scores = scores.transpose(1, 2)              # [B, T_out, 2]
        # scores = scores.squeeze(1)
        return scores


if __name__ == "__main__":
    # Example usage
    model = SleepEventDetector()
    # Create a dummy input tensor with shape (batch_size, channels, samples)
    input_tensor = torch.randn(8, 1, 5040)  # Batch size 8, 1 channel, 5120 samples
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Should be (8, T_out, 2) where T_out ~ 640