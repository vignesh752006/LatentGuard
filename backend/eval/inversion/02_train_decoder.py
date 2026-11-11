"""
Task 5 — Inversion Attack Benchmark: Stage 2 — Decoder Training
================================================================

Trains a small MLP decoder (embedding → 224×224 pixels) on V and V' separately.

Usage:
  pip install torch torchvision numpy pillow tqdm

  python backend/eval/inversion/02_train_decoder.py \\
      --data-dir /tmp/vdlm_eval \\
      --out-dir /tmp/vdlm_eval/models \\
      --sanitizer V_svd_k4 \\
      --epochs 50 \\
      --batch-size 16

Decoder architecture (MLP):
  Input:  [576 * 1152] = 663552 → too large; use PCA to 512 dims first
  Actual: mean-pooled [576, 1152] → [1152] → FC(1152, 2048) → FC(2048, 4096)
          → FC(4096, 224*224*3) → sigmoid → [224, 224, 3]

NOTE: Full training requires significant GPU compute (not run in CI).
      CI runs a sanity test with 10 samples, 2 epochs.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_dataset(data_dir: Path, sanitizer_key: str = "V", max_samples: int | None = None):
    """Load dataset from .npz files."""
    files = sorted(data_dir.glob("sample_*.npz"))
    if max_samples:
        files = files[:max_samples]

    Vs, screenshots = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        V = d[sanitizer_key] if sanitizer_key in d else d["V"]
        sc = d["screenshot"]
        Vs.append(V.astype(np.float32))
        screenshots.append(sc.astype(np.float32) / 255.0)

    return np.stack(Vs), np.stack(screenshots)


class MlpDecoder:
    """Minimal MLP decoder (numpy-only for portability)."""

    def __init__(self, input_dim: int = 1152, hidden: int = 2048, out_pixels: int = 224 * 224 * 3):
        rng = np.random.default_rng(0)
        self.W1 = rng.standard_normal((input_dim, hidden)).astype(np.float32) * 0.01
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden, out_pixels)).astype(np.float32) * 0.01
        self.b2 = np.zeros(out_pixels, dtype=np.float32)
        self.out_pixels = out_pixels

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: [B, D] → [B, out_pixels]"""
        h = np.maximum(0, x @ self.W1 + self.b1)
        out = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))
        return out

    def save(self, path: Path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: Path) -> "MlpDecoder":
        d = np.load(path)
        model = cls.__new__(cls)
        model.W1, model.b1, model.W2, model.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        model.out_pixels = model.W2.shape[1]
        return model


def train(
    data_dir: Path,
    out_dir: Path,
    sanitizer_key: str = "V",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    max_samples: int | None = None,
) -> dict:
    """Train the MLP decoder. Returns final metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Loading {sanitizer_key} from {data_dir}")
    Vs, targets = load_dataset(data_dir, sanitizer_key, max_samples)
    N = len(Vs)
    print(f"[train] {N} samples, V shape: {Vs.shape}")

    # Mean-pool embeddings: [N, 576, 1152] → [N, 1152]
    X = Vs.mean(axis=1)  # [N, 1152]
    Y = targets.reshape(N, -1)  # [N, 224*224*3]

    model = MlpDecoder(input_dim=X.shape[1], hidden=512, out_pixels=Y.shape[1])

    losses = []
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb, yb = X[batch_idx], Y[batch_idx]

            # Forward
            h_pre = xb @ model.W1 + model.b1
            h = np.maximum(0, h_pre)
            out_pre = h @ model.W2 + model.b2
            pred = 1 / (1 + np.exp(-out_pre))

            # MSE loss
            diff = pred - yb
            loss = (diff ** 2).mean()
            epoch_loss += loss
            n_batches += 1

            # Backward (chain rule)
            d_out = 2 * diff / diff.size * pred * (1 - pred)
            dW2 = h.T @ d_out
            db2 = d_out.sum(axis=0)
            d_h = d_out @ model.W2.T
            d_h_pre = d_h * (h_pre > 0).astype(np.float32)
            dW1 = xb.T @ d_h_pre
            db1 = d_h_pre.sum(axis=0)

            # SGD
            model.W2 -= lr * np.clip(dW2, -1, 1)
            model.b2 -= lr * np.clip(db2, -1, 1)
            model.W1 -= lr * np.clip(dW1, -1, 1)
            model.b1 -= lr * np.clip(db1, -1, 1)

        avg_loss = epoch_loss / max(1, n_batches)
        losses.append(float(avg_loss))
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[train] epoch {epoch+1}/{epochs} loss={avg_loss:.6f}")

    # Save model
    model_path = out_dir / f"decoder_{sanitizer_key}.npz"
    model.save(model_path)
    print(f"[train] Saved model to {model_path}")

    return {"sanitizer": sanitizer_key, "n_samples": N, "epochs": epochs, "final_loss": losses[-1]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sanitizer", default="V")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    result = train(
        Path(args.data_dir), Path(args.out_dir),
        args.sanitizer, args.epochs, args.batch_size,
        max_samples=args.max_samples,
    )
    print(json.dumps(result, indent=2))

