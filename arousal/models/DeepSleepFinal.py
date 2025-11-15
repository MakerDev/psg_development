import torch
import torch.nn as nn


class SpectrogramEncoder(nn.Module):
    """Encode spectrogram inputs while preserving the temporal resolution."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # down-sample frequency only

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(64, hidden_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, F, T)
        features = self.encoder(x)
        # Average frequency dimension -> (B, hidden_channels, T)
        features = torch.mean(features, dim=2)
        return features


class TimeFeatureEncoder(nn.Module):
    """Encode engineered time-domain features."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(64, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        return self.encoder(x)


class DeepSleepFinal(nn.Module):
    """Two-branch network that fuses spectral and time-domain evidence."""

    def __init__(
        self,
        n_spectrogram_channels: int,
        n_time_feature_channels: int,
        spec_hidden: int = 128,
        time_hidden: int = 64,
        fusion_hidden: int = 160,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.spec_encoder = SpectrogramEncoder(
            in_channels=n_spectrogram_channels,
            hidden_channels=spec_hidden,
            dropout=dropout / 2,
        )
        self.time_encoder = TimeFeatureEncoder(
            in_channels=n_time_feature_channels,
            hidden_channels=time_hidden,
            dropout=dropout / 2,
        )

        self.attention_gate = nn.Conv1d(time_hidden, spec_hidden, kernel_size=1)

        self.fusion = nn.Sequential(
            nn.Conv1d(spec_hidden + time_hidden, fusion_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(fusion_hidden, fusion_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Conv1d(fusion_hidden, 1, kernel_size=1)

    def forward(self, inputs, return_prob: bool = False):
        if not isinstance(inputs, dict):
            raise ValueError("DeepSleepFinal expects a dict with 'spectrogram' and 'time_features'.")

        spec = inputs["spectrogram"]  # (B, C, F, T)
        time_features = inputs["time_features"]  # (B, C, K, T)

        batch_size, channels, feature_bins, time_bins = time_features.shape
        time_features = time_features.reshape(batch_size, channels * feature_bins, time_bins)

        spec_repr = self.spec_encoder(spec)
        time_repr = self.time_encoder(time_features)

        gate = torch.sigmoid(self.attention_gate(time_repr))
        spec_repr = spec_repr * gate

        fused = torch.cat([spec_repr, time_repr], dim=1)
        fused = self.fusion(fused)
        logits = self.classifier(fused).squeeze(1)

        if return_prob:
            return torch.sigmoid(logits)
        return logits
