"""Run Task 5 dataset generation, decoder training, and evaluation end to end."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INVERSION_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = INVERSION_DIR / "runs" / "full"
SANITIZERS = [
    "V",
    "Vprime_none",
    "Vprime_rank1_alpha0",
    "Vprime_rank1_alpha001",
    "Vprime_svd_k1_alpha0",
    "Vprime_svd_k1_alpha001",
    "Vprime_svd_k4_alpha0",
    "Vprime_svd_k4_alpha001",
    "Vprime_svd_k8_alpha0",
    "Vprime_svd_k8_alpha001",
]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    parser.add_argument("--skip-ocr", action="store_true")
    args = parser.parse_args()
    data_dir = args.run_dir / "data"
    pages_dir = args.run_dir / "pages"
    models_dir = args.run_dir / "models"
    generator = INVERSION_DIR / "01_generate_dataset.py"
    trainer = INVERSION_DIR / "02_train_decoder.py"
    evaluator = INVERSION_DIR / "03_evaluate.py"

    run([
        sys.executable,
        str(generator),
        "--out-dir",
        str(data_dir),
        "--pages-dir",
        str(pages_dir),
        "--n",
        str(args.n),
    ])
    for sanitizer in SANITIZERS:
        run([
            sys.executable,
            str(trainer),
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(models_dir),
            "--sanitizer",
            sanitizer,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        ])
    command = [
        sys.executable,
        str(evaluator),
        "--data-dir",
        str(data_dir),
        "--model-dir",
        str(models_dir),
        "--out-dir",
        str(args.out_dir),
        "--sanitizers",
        *SANITIZERS,
    ]
    if args.skip_ocr:
        command.append("--skip-ocr")
    run(command)


if __name__ == "__main__":
    main()
