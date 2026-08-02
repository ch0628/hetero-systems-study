from __future__ import annotations

import argparse
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
from torchvision.models import ResNet18_Weights, resnet18

from benchmark import (
    benchmark,
    calculate_statistics,
    collect_environment,
    print_statistics,
    repo_relative,
    write_json,
)


def current_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


parser = argparse.ArgumentParser(description="PyTorch CPU inference benchmark")
parser.add_argument("--threads", type=int, required=True)
parser.add_argument("--batch", type=int, required=True, choices=[1, 4, 16])
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--iterations", type=int, default=300)
args = parser.parse_args()

if args.threads <= 0:
    raise ValueError("--threads must be > 0")

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT_DIR = BENCHMARK_DIR.parent
INPUT_DIR = BENCHMARK_DIR / "data" / "inputs"
RESULT_DIR = BENCHMARK_DIR / "results" / "pytorch"
RAW_DIR = BENCHMARK_DIR / "results" / "raw"
ENVIRONMENT_PATH = BENCHMARK_DIR / "results" / "environment" / "cpu_environment.json"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = INPUT_DIR / f"input_b{args.batch}.npy"
OUTPUT_PATH = RESULT_DIR / f"pytorch_output_b{args.batch}_t{args.threads}.npy"
METRICS_PATH = RESULT_DIR / f"pytorch_metrics_b{args.batch}_t{args.threads}.json"
RAW_PATH = RAW_DIR / f"pytorch_cpu_b{args.batch}_t{args.threads}_latencies.json"

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_PATH}\n"
        f"Run: python3 export_onnx.py --batch {args.batch}"
    )

SEED = 42
torch.manual_seed(SEED)
torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)

rss_before_model = current_rss_mb()
start = time.perf_counter()
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()
model_initialization_ms = (time.perf_counter() - start) * 1000.0
rss_after_model = current_rss_mb()
model_rss_delta_mb = rss_after_model - rss_before_model

start = time.perf_counter()
input_array = np.load(INPUT_PATH)
input_load_ms = (time.perf_counter() - start) * 1000.0

if input_array.ndim != 4:
    raise ValueError(f"Expected 4D input, got {input_array.shape}")
if input_array.shape[0] != args.batch:
    raise ValueError(f"Input batch mismatch: expected {args.batch}, got {input_array.shape[0]}")

start = time.perf_counter()
input_tensor = torch.from_numpy(input_array)
tensor_conversion_ms = (time.perf_counter() - start) * 1000.0

with torch.inference_mode():
    start = time.perf_counter()
    first_output = model(input_tensor)
    first_inference_ms = (time.perf_counter() - start) * 1000.0

pytorch_output_array = first_output.detach().cpu().numpy()
np.save(OUTPUT_PATH, pytorch_output_array)


def inference():
    with torch.inference_mode():
        return model(input_tensor)


times_ms = benchmark(
    inference_fn=inference,
    warming_count=args.warmup,
    measured_count=args.iterations,
)
stats = calculate_statistics(times_ms, args.batch)
print_statistics(stats)

final_rss_mb = current_rss_mb()
peak_memory_mb = peak_rss_mb()

run_metadata = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "command": " ".join(sys.argv),
    "runtime": "pytorch",
    "device": "cpu",
    "model": "resnet18",
    "weights": str(ResNet18_Weights.DEFAULT),
    "seed": SEED,
    "batch_size": args.batch,
    "input_shape": list(input_array.shape),
    "output_shape": list(pytorch_output_array.shape),
    "input_dtype": str(input_array.dtype),
    "intra_op_threads": torch.get_num_threads(),
    "inter_op_threads": torch.get_num_interop_threads(),
    "warmup_count": args.warmup,
    "measurement_count": args.iterations,
    "input_path": repo_relative(INPUT_PATH, ROOT_DIR),
    "output_path": repo_relative(OUTPUT_PATH, ROOT_DIR),
    "raw_latency_path": repo_relative(RAW_PATH, ROOT_DIR),
    "environment_path": repo_relative(ENVIRONMENT_PATH, ROOT_DIR),
}

raw_result = {
    **run_metadata,
    "latencies_ms": times_ms,
}

metrics = {
    **run_metadata,
    "input_load_ms": input_load_ms,
    "tensor_conversion_ms": tensor_conversion_ms,
    "model_initialization_ms": model_initialization_ms,
    "rss_before_model_mb": rss_before_model,
    "rss_after_model_mb": rss_after_model,
    "model_rss_delta_mb": model_rss_delta_mb,
    "first_inference_ms": first_inference_ms,
    **stats,
    "final_rss_mb": final_rss_mb,
    "peak_rss_mb": peak_memory_mb,
}

write_json(RAW_PATH, raw_result)
write_json(METRICS_PATH, metrics)
write_json(ENVIRONMENT_PATH, collect_environment(ROOT_DIR))

print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"Batch size: {args.batch}")
print(f"First inference: {first_inference_ms:.4f} ms")
print(f"Final RSS: {final_rss_mb:.2f} MB")
print(f"Peak RSS: {peak_memory_mb:.2f} MB")
print(f"Output saved: {OUTPUT_PATH}")
print(f"Raw latency saved: {RAW_PATH}")
print(f"Metrics saved: {METRICS_PATH}")
