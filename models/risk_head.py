
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedRiskHead(nn.Module):
    """
    Gated temporal pooling + risk regression + classification.
    """

    def __init__(self, temporal_dim: int, hidden_dim: int, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        self.gate_fc = nn.Linear(temporal_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

        self.reg_head = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: (B, T, C_temporal)

        Returns:
            y_reg: (B, 1)
            y_logits: (B, num_classes)
            gates: (B, T, 1)
        """
        gates = torch.sigmoid(self.gate_fc(h))  # (B, T, 1)
        weights = gates / (gates.sum(dim=1, keepdim=True) + 1e-6)
        r = (weights * h).sum(dim=1)  # (B, C_temporal)

        r = self.dropout(r)
        y_reg = self.reg_head(r)
        y_logits = self.cls_head(r)
        return y_reg, y_logits, gates
