from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from torchvision.models import ResNet18_Weights, resnet18

from benchmark import collect_environment, repo_relative, write_json


parser = argparse.ArgumentParser(description="Export fixed-shape ResNet18 ONNX models")
parser.add_argument("--batch", type=int, required=True, choices=[1, 4, 16])
parser.add_argument("--opset", type=int, default=17)
args = parser.parse_args()

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT_DIR = BENCHMARK_DIR.parent
INPUT_DIR = BENCHMARK_DIR / "data" / "inputs"
MODEL_DIR = BENCHMARK_DIR / "data" / "onnx_models"
EXPORT_RESULT_DIR = BENCHMARK_DIR / "results" / "onnx_export"
ENVIRONMENT_PATH = BENCHMARK_DIR / "results" / "environment" / "cpu_environment.json"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = INPUT_DIR / f"input_b{args.batch}.npy"
MODEL_PATH = MODEL_DIR / f"resnet18_b{args.batch}.onnx"
SUMMARY_PATH = EXPORT_RESULT_DIR / f"onnx_export_b{args.batch}.json"

SEED = 42
torch.manual_seed(SEED)

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

input_tensor = torch.randn(args.batch, 3, 224, 224, dtype=torch.float32)
np.save(INPUT_PATH, input_tensor.numpy())

start = time.perf_counter()
torch.onnx.export(
    model,
    input_tensor,
    MODEL_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=args.opset,
    do_constant_folding=True,
)
export_time_ms = (time.perf_counter() - start) * 1000.0

onnx_model = onnx.load(MODEL_PATH)
checker_passed = True
checker_error = None
try:
    onnx.checker.check_model(onnx_model)
except Exception as exc:  # checker failure should still be recorded
    checker_passed = False
    checker_error = str(exc)

operator_counts = dict(sorted(Counter(node.op_type for node in onnx_model.graph.node).items()))
model_size_bytes = MODEL_PATH.stat().st_size

summary = {
    "model": "resnet18",
    "weights": str(ResNet18_Weights.DEFAULT),
    "seed": SEED,
    "batch_size": args.batch,
    "fixed_shape": True,
    "input_name": "input",
    "output_name": "output",
    "input_shape": list(input_tensor.shape),
    "input_dtype": str(input_tensor.numpy().dtype),
    "opset_version": args.opset,
    "export_time_ms": export_time_ms,
    "model_size_bytes": model_size_bytes,
    "model_size_mb": model_size_bytes / (1024 ** 2),
    "checker_passed": checker_passed,
    "checker_error": checker_error,
    "graph_name": onnx_model.graph.name,
    "node_count": len(onnx_model.graph.node),
    "initializer_count": len(onnx_model.graph.initializer),
    "operator_counts": operator_counts,
    "input_path": repo_relative(INPUT_PATH, ROOT_DIR),
    "model_path": repo_relative(MODEL_PATH, ROOT_DIR),
}

write_json(SUMMARY_PATH, summary)
write_json(ENVIRONMENT_PATH, collect_environment(ROOT_DIR))

print(f"Batch size: {args.batch}")
print(f"Input shape: {tuple(input_tensor.shape)}")
print(f"Opset: {args.opset}")
print(f"Export time: {export_time_ms:.4f} ms")
print(f"ONNX size: {summary['model_size_mb']:.4f} MB")
print(f"ONNX checker: {'PASS' if checker_passed else 'FAIL'}")
print(f"Node count: {summary['node_count']}")
print(f"Input saved: {INPUT_PATH}")
print(f"ONNX model saved: {MODEL_PATH}")
print(f"Export summary saved: {SUMMARY_PATH}")

if not checker_passed:
    raise RuntimeError(f"ONNX checker failed: {checker_error}")
