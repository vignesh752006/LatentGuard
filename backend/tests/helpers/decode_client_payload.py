"""Decode a client-generated tensor payload for the cross-language test."""

import json
import sys

import numpy as np

from app.schemas.messages import TensorPayload
from app.tensor.serialization import TensorContract, decode_tensor_payload


raw = json.load(sys.stdin)
payload = TensorPayload.model_validate(raw["payload"])
contract = TensorContract(patch_count=576, embedding_dim=1152, dtype="float16")
decoded = decode_tensor_payload(payload, contract, max_elements=576 * 1152)

pattern = np.array([0.1, -0.2, 0.3, -1.1, 0.0, 2.1, 0.07], dtype=np.float32)
expected = np.resize(pattern, 576 * 1152).reshape(576, 1152)
max_abs_error = float(np.abs(decoded.numpy() - expected).max())
if max_abs_error > 1e-3:
    raise AssertionError(f"decoded tensor exceeded fp16 tolerance: {max_abs_error}")

print(json.dumps({"shape": list(decoded.shape), "max_abs_error": max_abs_error}))
