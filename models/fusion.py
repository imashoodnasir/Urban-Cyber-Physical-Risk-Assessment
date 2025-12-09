
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion operating on a set of modality feature maps.

    This module assumes that modality features have been stacked along the channel
    dimension and optionally split into modality groups outside of the module.
    """

    def __init__(self, in_dim: int, fused_dim: int, num_modalities: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.fused_dim = fused_dim
        self.num_modalities = num_modalities

        self.q_proj = nn.Linear(in_dim, fused_dim)
        self.k_proj = nn.Linear(in_dim, fused_dim)
        self.v_proj = nn.Linear(in_dim, fused_dim)

        self.out_proj = nn.Linear(fused_dim, fused_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, M, C) where M is number of modalities, C=in_dim

        Returns:
            fused: (B, T, C_fused)
        """
        B, T, M, C = x.shape
        x_flat = x.view(B * T, M, C)

        Q = self.q_proj(x_flat)  # (B*T, M, F)
        K = self.k_proj(x_flat)
        V = self.v_proj(x_flat)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.shape[-1] ** 0.5)  # (B*T, M, M)
        attn_weights = F.softmax(attn_scores, dim=-1)

        fused_modal = torch.bmm(attn_weights, V)  # (B*T, M, F)
        fused_modal = fused_modal.mean(dim=1)     # average over modalities: (B*T, F)
        fused_modal = self.out_proj(fused_modal)  # (B*T, F)

        fused = fused_modal.view(B, T, self.fused_dim)
        return fused
