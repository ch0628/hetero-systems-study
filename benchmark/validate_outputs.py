from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmark import repo_relative, write_json


parser = argparse.ArgumentParser(description="Validate PyTorch and ONNX Runtime CPU outputs")
parser.add_argument("--batch", type=int, required=True, choices=[1, 4, 16])
parser.add_argument("--threads", type=int, required=True)
parser.add_argument("--rtol", type=float, default=1e-4)
parser.add_argument("--atol", type=float, default=1e-5)
args = parser.parse_args()

if args.threads <= 0:
    raise ValueError("--threads must be > 0")

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT_DIR = BENCHMARK_DIR.parent
PYTORCH_OUTPUT_PATH = (
    BENCHMARK_DIR / "results" / "pytorch" / f"pytorch_output_b{args.batch}_t{args.threads}.npy"
)
ONNX_OUTPUT_PATH = (
    BENCHMARK_DIR / "results" / "onnx" / f"onnx_output_b{args.batch}_t{args.threads}.npy"
)
VALIDATION_PATH = (
    BENCHMARK_DIR / "results" / "validation" / f"validation_b{args.batch}_t{args.threads}.json"
)

for path in (PYTORCH_OUTPUT_PATH, ONNX_OUTPUT_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Output file not found: {path}")

pytorch_output = np.load(PYTORCH_OUTPUT_PATH)
onnx_output = np.load(ONNX_OUTPUT_PATH)

if pytorch_output.shape != onnx_output.shape:
    raise ValueError(
        f"Output shape mismatch: PyTorch={pytorch_output.shape}, ONNX={onnx_output.shape}"
    )

absolute_difference = np.abs(pytorch_output - onnx_output)
max_absolute_difference = float(np.max(absolute_difference))
mean_absolute_difference = float(np.mean(absolute_difference))
all_close = bool(np.allclose(pytorch_output, onnx_output, rtol=args.rtol, atol=args.atol))

pytorch_top1 = np.argmax(pytorch_output, axis=1)
onnx_top1 = np.argmax(onnx_output, axis=1)
same_top1 = bool(np.array_equal(pytorch_top1, onnx_top1))

result = {
    "batch_size": args.batch,
    "intra_op_threads": args.threads,
    "rtol": args.rtol,
    "atol": args.atol,
    "pytorch_output_path": repo_relative(PYTORCH_OUTPUT_PATH, ROOT_DIR),
    "onnx_output_path": repo_relative(ONNX_OUTPUT_PATH, ROOT_DIR),
    "pytorch_output_shape": list(pytorch_output.shape),
    "onnx_output_shape": list(onnx_output.shape),
    "max_absolute_difference": max_absolute_difference,
    "mean_absolute_difference": mean_absolute_difference,
    "outputs_all_close": all_close,
    "pytorch_top1": pytorch_top1.tolist(),
    "onnx_top1": onnx_top1.tolist(),
    "same_top1": same_top1,
}

write_json(VALIDATION_PATH, result)

print(f"Batch: {args.batch}")
print(f"Threads: {args.threads}")
print(f"Max absolute difference: {max_absolute_difference:.9g}")
print(f"Mean absolute difference: {mean_absolute_difference:.9g}")
print(f"Output all close: {all_close}")
print(f"Same Top-1: {same_top1}")
print(f"Validation saved: {VALIDATION_PATH}")

if not all_close or not same_top1:
    raise SystemExit(1)
