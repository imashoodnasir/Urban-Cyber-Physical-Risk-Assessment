
import argparse
import torch
from torch.utils.data import DataLoader

from config import Config
from datasets import RandomMultimodalDataset
from models import SpatioTemporalRiskModel
from utils import set_seed

def mc_dropout_predict(model, x, mc_steps: int):
    model.train()  # enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(mc_steps):
            y_reg, logits, _ = model(x)
            preds.append(y_reg.unsqueeze(0))
    preds = torch.cat(preds, dim=0)  # (S, B, 1)
    mean = preds.mean(dim=0)
    var = preds.var(dim=0)
    return mean, var

def run_inference(args):
    cfg = Config()
    set_seed(123)

    dataset = RandomMultimodalDataset(
        num_samples=args.num_samples,
        seq_len=cfg.seq_len,
        in_channels=cfg.in_channels,
        height=cfg.height,
        width=cfg.width,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalRiskModel(cfg).to(device)

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {args.checkpoint}")

    all_mean = []
    all_var = []

    for x, y_reg, y_cls in loader:
        x = x.to(device)
        mean, var = mc_dropout_predict(model, x, cfg.mc_steps)
        all_mean.append(mean.cpu())
        all_var.append(var.cpu())

    all_mean = torch.cat(all_mean, dim=0)
    all_var = torch.cat(all_var, dim=0)

    print("Predicted mean vulnerability (first 5):", all_mean[:5].view(-1).tolist())
    print("Predicted variance (first 5):", all_var[:5].view(-1).tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    run_inference(args)
