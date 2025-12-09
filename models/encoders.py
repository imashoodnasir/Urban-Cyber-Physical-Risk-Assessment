
import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    """
    Generic CNN encoder for raster modalities (optical, SAR, socioeconomic, traffic).
    """

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return self.net(x)


class SimpleGCNEncoder(nn.Module):
    """
    Very simple GCN-style encoder placeholder for OSM infrastructure graphs.
    This is intentionally minimal and should be replaced with a proper GCN in real use.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (B, N, F)
            adj_matrix: (B, N, N) adjacency matrix

        Returns:
            node_embeddings: (B, N, hidden_dim)
        """
        h = self.relu(self.fc1(node_features))
        h = torch.bmm(adj_matrix, h)  # simple neighborhood aggregation
        h = self.relu(self.fc2(h))
        return h
