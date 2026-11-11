#!/usr/bin/env python
"""Export the Hugging Face SigLIP vision tower as a checked FP16 ONNX graph."""

import argparse
import sys
from pathlib import Path

import onnx
import numpy as np
import torch
from onnxconverter_common import float16
from transformers import SiglipVisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.eval.siglip.projection import build_projection_matrix

MODEL_ID = "google/siglip-base-patch16-224"
OUTPUT_NAME = "siglip-base-patch16-224-vision-fp16.onnx"
ADAPTER_NAME = "siglip-base-patch16-224-adapter.onnx"
WASM_NAME = "siglip-base-patch16-224-vision-fp32.onnx"


class VisionTower(torch.nn.Module):
    def __init__(self, model: SiglipVisionModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).last_hidden_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "client" / "public" / "models"),
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    adapter_path = output_dir / ADAPTER_NAME
    wasm_path = output_dir / WASM_NAME
    fp32_path = wasm_path
    if output_path.exists() or adapter_path.exists() or wasm_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model files in {output_dir}")

    model = VisionTower(SiglipVisionModel.from_pretrained(args.model_id).eval().cpu().float()).eval()
    sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (sample,),
            str(fp32_path),
            input_names=["pixel_values"],
            output_names=["last_hidden_state"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    graph = onnx.load(str(fp32_path))
    graph = float16.convert_float_to_float16(graph, keep_io_types=True)
    graph_input = graph.graph.input[0]
    original_input = graph_input.name
    fp16_input = f"{original_input}_fp16"
    for node in graph.graph.node:
        for index, name in enumerate(node.input):
            if name == original_input:
                node.input[index] = fp16_input
    graph.graph.node.insert(
        0,
        onnx.helper.make_node(
            "Cast",
            [original_input],
            [fp16_input],
            to=onnx.TensorProto.FLOAT16,
            name="cast_pixel_values_to_fp16",
        ),
    )

    graph_output = graph.graph.output[0]
    original_output = graph_output.name
    fp16_output = f"{original_output}_fp16"
    output_producer = next(
        node for node in graph.graph.node if original_output in node.output
    )
    for index, name in enumerate(output_producer.output):
        if name == original_output:
            output_producer.output[index] = fp16_output
    graph.graph.node.append(
        onnx.helper.make_node(
            "Cast",
            [fp16_output],
            [original_output],
            to=onnx.TensorProto.FLOAT,
            name="cast_last_hidden_state_to_fp32",
        )
    )
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output_path))
    export_adapter(adapter_path)

    exported = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(exported)
    fp32_exported = onnx.load(str(wasm_path), load_external_data=False)
    onnx.checker.check_model(fp32_exported)
    input_shape = [d.dim_value for d in exported.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [d.dim_value for d in exported.graph.output[0].type.tensor_type.shape.dim]
    if input_shape != [1, 3, 224, 224] or output_shape != [1, 196, 768]:
        output_path.unlink()
        raise RuntimeError(f"Unexpected exported IO shapes: {input_shape} -> {output_shape}")
    print(f"Exported vision model {output_path} ({output_path.stat().st_size} bytes)")
    print(f"Exported WASM FP32 vision model {wasm_path} ({wasm_path.stat().st_size} bytes)")
    print(f"Exported fixed adapter {adapter_path} ({adapter_path.stat().st_size} bytes)")


def export_adapter(path: Path) -> None:
    import onnx.helper as helper
    from onnx import TensorProto

    def int64_initializer(name: str, values: list[int]) -> onnx.TensorProto:
        return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)

    initializers = [
        int64_initializer("grid_shape", [1, 14, 14, 768]),
        int64_initializer("target_size", [1, 768, 24, 24]),
        int64_initializer("flat_shape", [576, 768]),
        int64_initializer("output_shape", [1, 576, 1152]),
        helper.make_tensor(
            "projection",
            TensorProto.FLOAT,
            [768, 1152],
            build_projection_matrix().astype(np.float32).ravel().tolist(),
        ),
    ]
    nodes = [
        helper.make_node("Reshape", ["patch_embeddings", "grid_shape"], ["patch_grid"]),
        helper.make_node("Transpose", ["patch_grid"], ["patch_grid_nchw"], perm=[0, 3, 1, 2]),
        helper.make_node(
            "Resize",
            ["patch_grid_nchw", "", "", "target_size"],
            ["upsampled_nchw"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            nearest_mode="floor",
        ),
        helper.make_node("Transpose", ["upsampled_nchw"], ["upsampled_nhwc"], perm=[0, 2, 3, 1]),
        helper.make_node("Reshape", ["upsampled_nhwc", "flat_shape"], ["flat_patches"]),
        helper.make_node("MatMul", ["flat_patches", "projection"], ["projected"]),
        helper.make_node("Reshape", ["projected", "output_shape"], ["adapted_embeddings"]),
    ]
    graph = helper.make_graph(
        nodes,
        "siglip_patch_adapter",
        [helper.make_tensor_value_info("patch_embeddings", TensorProto.FLOAT, [1, 196, 768])],
        [helper.make_tensor_value_info("adapted_embeddings", TensorProto.FLOAT, [1, 576, 1152])],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="vdlm-siglip-adapter",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    main()
