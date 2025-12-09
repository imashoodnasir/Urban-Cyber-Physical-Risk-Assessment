
import torch
import torch.nn as nn

from .fusion import CrossModalAttentionFusion
from .temporal import SpatioTemporalEncoder
from .risk_head import GatedRiskHead
from .encoders import ConvEncoder

class SpatioTemporalRiskModel(nn.Module):
    """
    Simplified full model that stacks:
      - A generic ConvEncoder for all channels (for demonstration),
      - CrossModalAttentionFusion over synthetic modality dimension,
      - SpatioTemporalEncoder,
      - GatedRiskHead.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # For simplicity, assume a single ConvEncoder taking all channels.
        self.spatial_encoder = ConvEncoder(config.in_channels, config.spatial_hidden_dim)

        # We emulate M modalities by splitting channels; here we just reshape
        # into a fake "num_modalities" dimension for the fusion module.
        self.num_modalities = 4
        fused_input_dim = config.spatial_hidden_dim

        self.fusion = CrossModalAttentionFusion(
            in_dim=fused_input_dim,
            fused_dim=config.fused_dim,
            num_modalities=self.num_modalities,
        )

        self.spatio_temporal = SpatioTemporalEncoder(
            fused_dim=config.fused_dim,
            temporal_hidden_dim=config.temporal_hidden_dim,
            num_heads=config.temporal_heads,
            num_layers=config.temporal_layers,
        )

        self.risk_head = GatedRiskHead(
            temporal_dim=config.temporal_hidden_dim,
            hidden_dim=config.risk_hidden_dim,
            num_classes=config.num_classes,
            dropout_p=config.dropout_p,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C_in, H, W)
        """
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        spatial = self.spatial_encoder(x)  # (B*T, C_enc, H, W)
        spatial = spatial.mean(dim=[2, 3])  # (B*T, C_enc)

        spatial = spatial.view(B, T, 1, -1)  # pretend we have M=1 modality here
        spatial = spatial.expand(-1, -1, self.num_modalities, -1)  # (B, T, M, C_enc)

        fused = self.fusion(spatial)  # (B, T, C_fused)

        h = self.spatio_temporal(fused)  # (B, T, C_temporal)

        y_reg, y_logits, gates = self.risk_head(h)
        return y_reg, y_logits, gates
