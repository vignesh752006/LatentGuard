#!/usr/bin/env python
"""Export SigLIP's text tower, local tokenizer, and sanitized-patch score adapter."""

import argparse
from pathlib import Path

import onnx
import numpy as np
import torch
from onnxconverter_common import float16
from tokenizers import processors
from transformers import SiglipModel, SiglipTokenizer
from transformers.convert_slow_tokenizer import SpmConverter

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.eval.siglip.projection import build_projection_matrix

MODEL_ID = "google/siglip-base-patch16-224"
FP32_NAME = "siglip-base-patch16-224-text-fp32.onnx"
FP16_NAME = "siglip-base-patch16-224-text-fp16.onnx"
TOKENIZER_DIR = "siglip-base-patch16-224-text-tokenizer"
SCORE_ADAPTER_NAME = "siglip-base-patch16-224-text-image-score.onnx"
SOURCE_SCORE_ADAPTER_NAME = "siglip-base-patch16-224-source-text-score.onnx"
TEXT_LENGTH = 64


class SiglipTokenizerConverter(SpmConverter):
    handle_byte_fallback = True


class TextTower(torch.nn.Module):
    def __init__(self, model: SiglipModel):
        super().__init__()
        self.text_model = model.text_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output


class PatchScoreAdapter(torch.nn.Module):
    def __init__(self, mapping: np.ndarray):
        super().__init__()
        self.register_buffer(
            "adapter_to_siglip_space",
            torch.from_numpy(mapping.astype(np.float32)),
        )

    def forward(
        self,
        adapted_embeddings: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        image_embedding = torch.matmul(
            adapted_embeddings,
            self.adapter_to_siglip_space,
        )
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)
        text_embedding = torch.nn.functional.normalize(text_embedding, dim=-1)
        return torch.sum(image_embedding * text_embedding[:, None, :], dim=-1)


class SourcePatchScoreAdapter(torch.nn.Module):
    def forward(
        self,
        source_embeddings: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        grid = source_embeddings.reshape(1, 14, 14, 768).permute(0, 3, 1, 2)
        grid = torch.nn.functional.interpolate(
            grid,
            size=(24, 24),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = grid.permute(0, 2, 3, 1).reshape(1, 576, 768)
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)
        text_embedding = torch.nn.functional.normalize(text_embedding, dim=-1)
        return torch.sum(image_embedding * text_embedding[:, None, :], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "client" / "public" / "models",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()
    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / FP32_NAME
    fp16_path = output_dir / FP16_NAME
    score_adapter_path = output_dir / SCORE_ADAPTER_NAME
    source_score_adapter_path = output_dir / SOURCE_SCORE_ADAPTER_NAME
    tokenizer_dir = output_dir / TOKENIZER_DIR
    if (
        fp32_path.exists()
        or fp16_path.exists()
        or score_adapter_path.exists()
        or source_score_adapter_path.exists()
        or tokenizer_dir.exists()
    ):
        raise FileExistsError(
            f"Refusing to overwrite existing SigLIP text assets in {output_dir}"
        )

    slow_tokenizer = SiglipTokenizer.from_pretrained(args.model_id)
    slow_tokenizer.save_pretrained(tokenizer_dir)
    fast_tokenizer = SiglipTokenizerConverter(slow_tokenizer).converted()
    eos_token = str(slow_tokenizer.eos_token)
    fast_tokenizer.post_processor = processors.TemplateProcessing(
        single=f"$A {eos_token}",
        special_tokens=[(eos_token, slow_tokenizer.eos_token_id)],
    )
    fast_tokenizer.enable_padding(
        length=TEXT_LENGTH,
        pad_id=slow_tokenizer.pad_token_id,
        pad_token=str(slow_tokenizer.pad_token),
    )
    fast_tokenizer.enable_truncation(max_length=TEXT_LENGTH)
    for sample in (
        "click the sign in button",
        "enter the account password",
        "find the billing address field",
        "open the online banking dashboard",
        "cancel the checkout and return to the cart",
    ):
        expected = slow_tokenizer(
            sample,
            padding="max_length",
            truncation=True,
            max_length=TEXT_LENGTH,
            return_attention_mask=True,
        )["input_ids"]
        actual = fast_tokenizer.encode(sample).ids
        if actual != expected:
            raise RuntimeError(
                f"SigLIP browser tokenizer does not match the reference for {sample!r}"
            )
    fast_tokenizer.save(str(tokenizer_dir / "tokenizer.json"))

    full_model = SiglipModel.from_pretrained(args.model_id).eval().cpu().float()
    model = TextTower(full_model).eval()
    input_ids = torch.zeros((1, TEXT_LENGTH), dtype=torch.long)
    attention_mask = torch.ones((1, TEXT_LENGTH), dtype=torch.long)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embedding"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    fp32_graph = onnx.load(str(fp32_path))
    onnx.checker.check_model(fp32_graph)
    fp16_graph = float16.convert_float_to_float16(fp32_graph, keep_io_types=True)
    onnx.checker.check_model(fp16_graph)
    onnx.save(fp16_graph, str(fp16_path))
    for path in (fp32_path, fp16_path):
        session = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(session)

    projection = build_projection_matrix().astype(np.float64)
    pseudo_inverse = projection.T @ np.linalg.inv(projection @ projection.T)
    mapping = pseudo_inverse
    identity_error = np.max(np.abs(projection @ pseudo_inverse - np.eye(768)))
    if identity_error > 1e-4:
        raise RuntimeError(
            f"Fixed projection pseudo-inverse residual is too large: {identity_error}"
        )
    score_adapter = PatchScoreAdapter(mapping).eval()
    torch.onnx.export(
        score_adapter,
        (
            torch.zeros((1, 576, 1152), dtype=torch.float32),
            torch.zeros((1, 768), dtype=torch.float32),
        ),
        str(score_adapter_path),
        input_names=["adapted_embeddings", "text_embedding"],
        output_names=["score_grid"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    score_graph = onnx.load(str(score_adapter_path))
    onnx.checker.check_model(score_graph)
    source_score_adapter = SourcePatchScoreAdapter().eval()
    torch.onnx.export(
        source_score_adapter,
        (
            torch.zeros((1, 196, 768), dtype=torch.float32),
            torch.zeros((1, 768), dtype=torch.float32),
        ),
        str(source_score_adapter_path),
        input_names=["source_embeddings", "text_embedding"],
        output_names=["score_grid"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    source_score_graph = onnx.load(str(source_score_adapter_path))
    onnx.checker.check_model(source_score_graph)
    print(f"Exported FP32 text tower: {fp32_path} ({fp32_path.stat().st_size} bytes)")
    print(f"Exported FP16 text tower: {fp16_path} ({fp16_path.stat().st_size} bytes)")
    print(
        f"Exported SigLIP image-text score adapter: {score_adapter_path} "
        f"({score_adapter_path.stat().st_size} bytes; pseudo-inverse residual {identity_error:.2e})"
    )
    print(
        f"Exported SigLIP source-grid score adapter: {source_score_adapter_path} "
        f"({source_score_adapter_path.stat().st_size} bytes)"
    )
    print(f"Saved tokenizer files: {tokenizer_dir}")


if __name__ == "__main__":
    main()
