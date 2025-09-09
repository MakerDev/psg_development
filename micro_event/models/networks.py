import torch
import torch.nn as nn
from .layers import MultiDilatedConv1D, CWTLayer

class REDv2Time(nn.Module):
    """REDv2-Time model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initial normalization
        self.input_norm = nn.BatchNorm1d(1)
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, config.stem_filters, 3, padding=1),
            nn.BatchNorm1d(config.stem_filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(config.stem_filters, config.stem_filters, 3, padding=1),
            nn.BatchNorm1d(config.stem_filters),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2)
        )
        
        # Multi-dilated convolutions
        self.mdconv1 = MultiDilatedConv1D(config.stem_filters, config.stem_filters * 2)
        self.pool1 = nn.AvgPool1d(2)
        
        self.mdconv2 = MultiDilatedConv1D(config.stem_filters * 2, config.stem_filters * 4)
        self.pool2 = nn.AvgPool1d(2)
        
        # LSTM
        self.lstm = nn.LSTM(
            config.stem_filters * 4,
            config.lstm_size,
            bidirectional=True,
            batch_first=True
        )
        
        # Classification head
        lstm_out_size = config.lstm_size * 2
        
        if config.fc_units > 0:
            self.fc = nn.Sequential(
                nn.Dropout(config.dropout_rate),
                nn.Conv1d(lstm_out_size, config.fc_units, 1),
                nn.ReLU(inplace=True)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout_rate),
                nn.Conv1d(config.fc_units, 2, 1)
            )
        else:
            self.fc = None
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout_rate),
                nn.Conv1d(lstm_out_size, 2, 1)
            )
    
    def forward(self, x):
        # Input shape: [batch, time]
        x = x.unsqueeze(1)  # [batch, 1, time]
        
        # Normalize
        x = self.input_norm(x)
        
        # Stem
        x = self.stem(x)
        
        # Multi-dilated convolutions
        x = self.mdconv1(x)
        x = self.pool1(x)
        
        x = self.mdconv2(x)
        x = self.pool2(x)
        
        # Prepare for LSTM: [batch, channels, time] -> [batch, time, channels]
        x = x.transpose(1, 2)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Back to conv format: [batch, time, channels] -> [batch, channels, time]
        x = x.transpose(1, 2)
        
        # Classification
        if self.fc is not None:
            x = self.fc(x)
        
        logits = self.classifier(x)
        
        return logits

class REDv2CWT1D(nn.Module):
    """REDv2-CWT1D model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # CWT layer
        self.cwt = CWTLayer(
            fs=config.fs,
            n_scales=32,
            lower_freq=0.5,
            upper_freq=30,
            fb_list=[1.5]
        )
        
        # Stem (flattened CWT features)
        cwt_channels = 32 * 2  # n_scales * 2 (real + imag)
        self.stem = nn.Sequential(
            nn.Conv1d(cwt_channels, config.stem_filters, 3, padding=1),
            nn.BatchNorm1d(config.stem_filters),
            nn.ReLU(inplace=True)
        )
        
        # Multi-dilated convolutions
        self.mdconv1 = MultiDilatedConv1D(config.stem_filters, config.stem_filters * 2)
        self.pool1 = nn.AvgPool1d(2)
        
        self.mdconv2 = MultiDilatedConv1D(config.stem_filters * 2, config.stem_filters * 4)
        self.pool2 = nn.AvgPool1d(2)
        
        # LSTM
        self.lstm = nn.LSTM(
            config.stem_filters * 4,
            config.lstm_size,
            bidirectional=True,
            batch_first=True
        )
        
        # Classification head
        lstm_out_size = config.lstm_size * 2
        
        if config.fc_units > 0:
            self.fc = nn.Sequential(
                nn.Dropout(config.dropout_rate),
                nn.Conv1d(lstm_out_size, config.fc_units, 1),
                nn.ReLU(inplace=True)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout_rate),
                nn.Conv1d(config.fc_units, 2, 1)
            )
        else:
            self.fc = None
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout_rate),
                nn.Conv1d(lstm_out_size, 2, 1)
            )
    
    def forward(self, x):
        # Input shape: [batch, time]
        
        # Apply CWT: [batch, time] -> [batch, time, scales, channels]
        cwt = self.cwt(x)
        
        # Flatten CWT features: [batch, time, scales, channels] -> [batch, time, scales*channels]
        batch_size, time_len, n_scales, n_channels = cwt.shape
        cwt = cwt.reshape(batch_size, time_len, -1)
        
        # Transpose for conv: [batch, time, features] -> [batch, features, time]
        x = cwt.transpose(1, 2)
        
        # Stem
        x = self.stem(x)
        
        # Multi-dilated convolutions
        x = self.mdconv1(x)
        x = self.pool1(x)
        
        x = self.mdconv2(x)
        x = self.pool2(x)
        
        # Prepare for LSTM: [batch, channels, time] -> [batch, time, channels]
        x = x.transpose(1, 2)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Back to conv format: [batch, time, channels] -> [batch, channels, time]
        x = x.transpose(1, 2)
        
        # Classification
        if self.fc is not None:
            x = self.fc(x)
        
        logits = self.classifier(x)
        
        return logits

def create_model(config):
    """Create model based on config."""
    if config.model_version == "v2_time":
        return REDv2Time(config)
    elif config.model_version == "v2_cwt1d":
        return REDv2CWT1D(config)
    else:
        raise ValueError(f"Unknown model version: {config.model_version}")