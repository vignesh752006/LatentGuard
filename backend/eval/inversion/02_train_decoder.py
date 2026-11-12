"""Train the convolutional latent-to-image inversion decoder."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from backend.eval.inversion.decoder import ConvDecoder


class LatentPageDataset(Dataset):
    def __init__(self, data_dir: Path, sanitizer_key: str, split: str):
        manifest_path = data_dir / "dataset-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = [
            data_dir / f"sample_{index:04d}.npz"
            for index in range(manifest["page_count"])
        ]
        feature_key = "V" if sanitizer_key == "V" else sanitizer_key
        self.records: list[tuple[np.ndarray, np.ndarray]] = []
        for file_path in files:
            with np.load(file_path, allow_pickle=False) as record:
                metadata = json.loads(str(record["metadata"].item()))
                if metadata["train_val_split"] != split:
                    continue
                latent = record[feature_key].astype(np.float32, copy=False)
                screenshot = record["screenshot"].astype(np.float32) / 255.0
            latent = latent.reshape(24, 24, 1152).transpose(2, 0, 1)
            target = screenshot.transpose(2, 0, 1)
            self.records.append((latent, target))
        if not self.records:
            raise ValueError(f"No {split} samples found in {data_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        latent, target = self.records[item]
        return torch.from_numpy(latent), torch.from_numpy(target)


def train(
    data_dir: Path,
    out_dir: Path,
    sanitizer_key: str = "V",
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
) -> dict[str, object]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(26171)
    np.random.seed(26171)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = LatentPageDataset(data_dir, sanitizer_key, "train")
    validation_dataset = LatentPageDataset(data_dir, sanitizer_key, "validation")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = ConvDecoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for latent, target in train_loader:
            latent, target = latent.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(latent), target)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            train_batches += 1
        model.eval()
        validation_loss = 0.0
        validation_batches = 0
        with torch.no_grad():
            for latent, target in validation_loader:
                latent, target = latent.to(device), target.to(device)
                validation_loss += float(F.mse_loss(model(latent), target).cpu())
                validation_batches += 1
        history.append({
            "epoch": epoch + 1,
            "train_mse": train_loss / max(1, train_batches),
            "validation_mse": validation_loss / max(1, validation_batches),
        })
        print(
            f"[train:{sanitizer_key}] epoch={epoch + 1}/{epochs} "
            f"train_mse={history[-1]['train_mse']:.6f} "
            f"validation_mse={history[-1]['validation_mse']:.6f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"decoder_{sanitizer_key}.pt"
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "sanitizer_key": sanitizer_key,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": 26171,
        "architecture": "conv1x1-1152x64+transposed-conv-24-to-192",
    }, model_path)
    result = {
        "sanitizer": sanitizer_key,
        "training_pages": len(train_dataset),
        "validation_pages": len(validation_dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "model": str(model_path),
        "history": history,
    }
    (out_dir / f"training_{sanitizer_key}.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sanitizer", default="V")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = train(
        args.data_dir,
        args.out_dir,
        args.sanitizer,
        args.epochs,
        args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
