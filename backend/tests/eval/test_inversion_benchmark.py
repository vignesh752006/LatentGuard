import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "client" / "public" / "models" / "siglip-base-patch16-224-vision-fp32.onnx"
GENERATOR = ROOT / "backend" / "eval" / "inversion" / "01_generate_dataset.py"
TRAINER = ROOT / "backend" / "eval" / "inversion" / "02_train_decoder.py"
EVALUATOR = ROOT / "backend" / "eval" / "inversion" / "03_evaluate.py"


def _run(*args: str) -> None:
    subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=True,
        timeout=900,
    )


def test_mini_inversion_pipeline_10_pages_2_epochs(tmp_path: Path):
    if not MODEL_PATH.is_file():
        if os.environ.get("CI", "").lower() == "true":
            pytest.fail(f"CI requires the real SigLIP model for inversion benchmark: {MODEL_PATH}")
        pytest.skip("local SigLIP FP32 ONNX model is not installed")

    data_dir = tmp_path / "data"
    pages_dir = tmp_path / "pages"
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "report"
    _run(
        str(GENERATOR),
        "--out-dir",
        str(data_dir),
        "--pages-dir",
        str(pages_dir),
        "--n",
        "10",
        "--seed",
        "9001",
    )
    assert len(list(data_dir.glob("sample_*.npz"))) == 10

    keys = ["V", "Vprime_svd_k4_alpha001"]
    for key in keys:
        _run(
            str(TRAINER),
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(model_dir),
            "--sanitizer",
            key,
            "--epochs",
            "2",
            "--batch-size",
            "2",
        )

    _run(
        str(EVALUATOR),
        "--data-dir",
        str(data_dir),
        "--model-dir",
        str(model_dir),
        "--out-dir",
        str(report_dir),
        "--sanitizers",
        *keys,
        "--skip-ocr",
    )
    result = json.loads((report_dir / "eval_inversion_results.json").read_text(encoding="utf-8"))
    assert result["setup"]["pages"] == 10
    assert result["setup"]["held_out_pages"] == 2
    assert set(result["results"]) == set(keys)
    assert (report_dir / "EVAL_INVERSION.md").is_file()
    assert (report_dir / "inversion-psnr.png").is_file()
