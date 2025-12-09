
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x.transpose(1, 2)  # (B, T, hidden_dim)
        return x


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        return self.encoder(x)


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, fused_dim: int, temporal_hidden_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.temp_conv = TemporalConvBlock(fused_dim, temporal_hidden_dim)
        self.transformer = TemporalTransformerEncoder(
            hidden_dim=temporal_hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C_fused)

        Returns:
            h: (B, T, C_temporal)
        """
        h = self.temp_conv(x)
        h = self.transformer(h)
        return h
