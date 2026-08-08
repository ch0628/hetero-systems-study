from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare saved PyTorch and ONNX Runtime outputs"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--batch", type=int, required=True, choices=[1, 4, 16])
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--pytorch-output", type=Path)
    parser.add_argument("--onnx-output", type=Path)
    parser.add_argument("--validation-id", default="latest")
    args = parser.parse_args()

    benchmark_dir = Path(__file__).resolve().parent
    pytorch_dir = benchmark_dir / "results" / "pytorch"
    onnx_dir = benchmark_dir / "results" / "onnx"
    validation_dir = benchmark_dir / "results" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    pytorch_path = args.pytorch_output or (
        pytorch_dir / f"pytorch_output_{args.device}_b{args.batch}.npy"
    )
    onnx_path = args.onnx_output or (
        onnx_dir / f"onnx_output_{args.device}_b{args.batch}.npy"
    )
    if not pytorch_path.exists():
        raise FileNotFoundError(f"PyTorch output not found: {pytorch_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX output not found: {onnx_path}")

    pytorch_output = np.load(pytorch_path)
    onnx_output = np.load(onnx_path)
    if pytorch_output.shape != onnx_output.shape:
        raise ValueError(
            f"Output shape mismatch: PyTorch={pytorch_output.shape}, "
            f"ONNX={onnx_output.shape}"
        )

    absolute_difference = np.abs(pytorch_output - onnx_output)
    pytorch_top1 = np.argmax(pytorch_output, axis=1)
    onnx_top1 = np.argmax(onnx_output, axis=1)
    top1_equal = pytorch_top1 == onnx_top1

    result = {
        "device": args.device,
        "batch_size": args.batch,
        "validation_id": args.validation_id,
        "pytorch_output_path": str(pytorch_path),
        "onnx_output_path": str(onnx_path),
        "output_shape": list(pytorch_output.shape),
        "output_dtype_pytorch": str(pytorch_output.dtype),
        "output_dtype_onnx": str(onnx_output.dtype),
        "rtol": args.rtol,
        "atol": args.atol,
        "max_absolute_difference": float(np.max(absolute_difference)),
        "mean_absolute_difference": float(np.mean(absolute_difference)),
        "outputs_all_close": bool(
            np.allclose(
                pytorch_output,
                onnx_output,
                rtol=args.rtol,
                atol=args.atol,
            )
        ),
        "same_top1": bool(np.all(top1_equal)),
        "top1_match_count": int(np.sum(top1_equal)),
        "top1_total_count": int(top1_equal.size),
        "pytorch_top1": pytorch_top1.tolist(),
        "onnx_top1": onnx_top1.tolist(),
    }

    validation_path = (
        validation_dir
        / f"validation_{args.device}_b{args.batch}_{args.validation_id}.json"
    )
    validation_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("PyTorch output:", pytorch_path)
    print("ONNX output:", onnx_path)
    print("Output shape:", pytorch_output.shape)
    print("Max absolute difference:", result["max_absolute_difference"])
    print("Mean absolute difference:", result["mean_absolute_difference"])
    print("Output all close:", result["outputs_all_close"])
    print("Same Top-1:", result["same_top1"])
    print("Validation saved:", validation_path)


if __name__ == "__main__":
    main()
