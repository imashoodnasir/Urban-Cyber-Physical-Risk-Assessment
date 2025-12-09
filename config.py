
import dataclasses

@dataclasses.dataclass
class Config:
    in_channels: int = 16
    height: int = 32
    width: int = 32
    seq_len: int = 10

    spatial_hidden_dim: int = 64
    fused_dim: int = 128
    temporal_hidden_dim: int = 128
    temporal_heads: int = 4
    temporal_layers: int = 2

    risk_hidden_dim: int = 128
    num_classes: int = 3

    lr: float = 1e-4
    weight_decay: float = 5e-4
    batch_size: int = 4
    epochs: int = 5

    dropout_p: float = 0.3
    mc_steps: int = 30
