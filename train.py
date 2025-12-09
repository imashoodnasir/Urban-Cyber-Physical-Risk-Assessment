
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from datasets import RandomMultimodalDataset
from models import SpatioTemporalRiskModel
from utils import set_seed, mae, rmse

def train(args):
    cfg = Config()
    set_seed(123)

    dataset = RandomMultimodalDataset(
        num_samples=args.num_samples,
        seq_len=cfg.seq_len,
        in_channels=cfg.in_channels,
        height=cfg.height,
        width=cfg.width,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalRiskModel(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for x, y_reg, y_cls in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            x = x.to(device)  # (B, T, C, H, W)
            y_reg = y_reg.to(device).float()
            y_cls = y_cls.to(device).long().view(-1)

            optimizer.zero_grad()
            pred_reg, logits, _ = model(x)

            reg_loss = F.mse_loss(pred_reg.view_as(y_reg), y_reg)
            cls_loss = F.cross_entropy(logits, y_cls)
            loss = reg_loss + 0.5 * cls_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}")

    # Simple evaluation on the same dataset
    model.eval()
    with torch.no_grad():
        all_reg = []
        all_reg_target = []
        for x, y_reg, y_cls in DataLoader(dataset, batch_size=cfg.batch_size):
            x = x.to(device)
            y_reg = y_reg.to(device).float()
            pred_reg, logits, _ = model(x)
            all_reg.append(pred_reg.cpu())
            all_reg_target.append(y_reg.cpu())
        all_reg = torch.cat(all_reg, dim=0)
        all_reg_target = torch.cat(all_reg_target, dim=0)
        print("MAE:", mae(all_reg, all_reg_target))
        print("RMSE:", rmse(all_reg, all_reg_target))

    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--save-path", type=str, default="model.pt")
    args = parser.parse_args()
    train(args)
