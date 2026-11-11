#!/usr/bin/env sh
set -eu
python "$(dirname "$0")/download-siglip-onnx.py" "$@"
