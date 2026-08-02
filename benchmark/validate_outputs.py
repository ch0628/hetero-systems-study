import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(
    description="Compare PyTorch and ONNX Runtime outputs"
)

parser.add_argument(
    "--device",
    choices=["cpu", "cuda"],
    default="cpu",
)

parser.add_argument(
    "--batch",
    type=int,
    required=True,
    choices=[1, 4, 16],
)

args = parser.parse_args()


BENCHMARK_DIR = Path(__file__).resolve().parent

PYTORCH_RESULT_DIR = (
    BENCHMARK_DIR / "results" / "pytorch"
)

ONNX_RESULT_DIR = (
    BENCHMARK_DIR / "results" / "onnx"
)

VALIDATION_RESULT_DIR = (
    BENCHMARK_DIR / "results" / "validation"
)

VALIDATION_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if args.device == "cpu":
    PYTORCH_OUTPUT_PATH = (
        PYTORCH_RESULT_DIR
        / f"pytorch_output_b{args.batch}.npy"
    )

    ONNX_OUTPUT_PATH = (
        ONNX_RESULT_DIR
        / f"onnx_output_b{args.batch}.npy"
    )

    VALIDATION_PATH = (
        VALIDATION_RESULT_DIR
        / f"validation_b{args.batch}.json"
    )
else:
    PYTORCH_OUTPUT_PATH = (
        PYTORCH_RESULT_DIR
        / f"pytorch_output_cuda_b{args.batch}.npy"
    )

    ONNX_OUTPUT_PATH = (
        ONNX_RESULT_DIR
        / f"onnx_output_cuda_b{args.batch}.npy"
    )

    VALIDATION_PATH = (
        VALIDATION_RESULT_DIR
        / f"validation_cuda_b{args.batch}.json"
    )


if not PYTORCH_OUTPUT_PATH.exists():
    raise FileNotFoundError(
        f"PyTorch output not found: "
        f"{PYTORCH_OUTPUT_PATH}"
    )

if not ONNX_OUTPUT_PATH.exists():
    raise FileNotFoundError(
        f"ONNX output not found: "
        f"{ONNX_OUTPUT_PATH}"
    )


pytorch_output = np.load(PYTORCH_OUTPUT_PATH)
onnx_output = np.load(ONNX_OUTPUT_PATH)


if pytorch_output.shape != onnx_output.shape:
    raise ValueError(
        f"Output shape mismatch: "
        f"PyTorch={pytorch_output.shape}, "
        f"ONNX={onnx_output.shape}"
    )


absolute_difference = np.abs(
    pytorch_output - onnx_output
)

max_absolute_difference = float(
    np.max(absolute_difference)
)

mean_absolute_difference = float(
    np.mean(absolute_difference)
)

outputs_all_close = bool(
    np.allclose(
        pytorch_output,
        onnx_output,
        rtol=1e-4,
        atol=1e-5,
    )
)

pytorch_top1 = np.argmax(
    pytorch_output,
    axis=1,
)

onnx_top1 = np.argmax(
    onnx_output,
    axis=1,
)

same_top1 = bool(
    np.array_equal(
        pytorch_top1,
        onnx_top1,
    )
)


print("Device:", args.device)
print("Batch:", args.batch)
print("PyTorch shape:", pytorch_output.shape)
print("ONNX shape:", onnx_output.shape)
print(f"Max difference: {max_absolute_difference:.8e}")
print(f"Mean difference: {mean_absolute_difference:.8e}")
print("Output all close:", outputs_all_close)
print("Same Top-1:", same_top1)


result = {
    "device": args.device,
    "batch_size": args.batch,
    "pytorch_output_path": str(PYTORCH_OUTPUT_PATH),
    "onnx_output_path": str(ONNX_OUTPUT_PATH),
    "output_shape": list(pytorch_output.shape),
    "max_absolute_difference": max_absolute_difference,
    "mean_absolute_difference": mean_absolute_difference,
    "rtol": 1e-4,
    "atol": 1e-5,
    "outputs_all_close": outputs_all_close,
    "pytorch_top1": pytorch_top1.tolist(),
    "onnx_top1": onnx_top1.tolist(),
    "same_top1": same_top1,
}


with VALIDATION_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        result,
        file,
        indent=2,
        ensure_ascii=False,
    )

print("Validation saved:", VALIDATION_PATH)
