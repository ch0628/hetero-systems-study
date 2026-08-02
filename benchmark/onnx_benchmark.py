from __future__ import annotations

import argparse
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil

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


parser = argparse.ArgumentParser(description="ONNX Runtime CPU inference benchmark")
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
MODEL_DIR = BENCHMARK_DIR / "data" / "onnx_models"
RESULT_DIR = BENCHMARK_DIR / "results" / "onnx"
RAW_DIR = BENCHMARK_DIR / "results" / "raw"
ENVIRONMENT_PATH = BENCHMARK_DIR / "results" / "environment" / "cpu_environment.json"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = INPUT_DIR / f"input_b{args.batch}.npy"
MODEL_PATH = MODEL_DIR / f"resnet18_b{args.batch}.onnx"
OUTPUT_PATH = RESULT_DIR / f"onnx_output_b{args.batch}_t{args.threads}.npy"
METRICS_PATH = RESULT_DIR / f"onnx_metrics_b{args.batch}_t{args.threads}.json"
RAW_PATH = RAW_DIR / f"onnx_cpu_b{args.batch}_t{args.threads}_latencies.json"

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_PATH}\n"
        f"Run: python3 export_onnx.py --batch {args.batch}"
    )
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}\n"
        f"Run: python3 export_onnx.py --batch {args.batch}"
    )

start = time.perf_counter()
input_array = np.load(INPUT_PATH)
input_load_ms = (time.perf_counter() - start) * 1000.0

if input_array.ndim != 4:
    raise ValueError(f"Expected 4D input, got {input_array.shape}")
if input_array.shape[0] != args.batch:
    raise ValueError(f"Input batch mismatch: expected {args.batch}, got {input_array.shape[0]}")

options = ort.SessionOptions()
options.intra_op_num_threads = args.threads
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

rss_before_session = current_rss_mb()
start = time.perf_counter()
session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)
session_load_ms = (time.perf_counter() - start) * 1000.0
rss_after_session = current_rss_mb()
session_rss_delta_mb = rss_after_session - rss_before_session

model_input = session.get_inputs()[0]
model_output = session.get_outputs()[0]
input_name = model_input.name

start = time.perf_counter()
first_outputs = session.run(None, {input_name: input_array})
first_inference_ms = (time.perf_counter() - start) * 1000.0
first_output = first_outputs[0]
np.save(OUTPUT_PATH, first_output)


def inference():
    return session.run(None, {input_name: input_array})


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
    "runtime": "onnxruntime",
    "provider": "CPUExecutionProvider",
    "available_providers": ort.get_available_providers(),
    "session_providers": session.get_providers(),
    "model": "resnet18",
    "batch_size": args.batch,
    "input_shape": list(input_array.shape),
    "output_shape": list(first_output.shape),
    "input_dtype": str(input_array.dtype),
    "intra_op_threads": args.threads,
    "inter_op_threads": 1,
    "execution_mode": "ORT_SEQUENTIAL",
    "warmup_count": args.warmup,
    "measurement_count": args.iterations,
    "input_name": model_input.name,
    "model_input_shape": model_input.shape,
    "model_input_type": model_input.type,
    "output_name": model_output.name,
    "model_output_shape": model_output.shape,
    "model_output_type": model_output.type,
    "input_path": repo_relative(INPUT_PATH, ROOT_DIR),
    "model_path": repo_relative(MODEL_PATH, ROOT_DIR),
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
    "session_load_ms": session_load_ms,
    "rss_before_session_mb": rss_before_session,
    "rss_after_session_mb": rss_after_session,
    "session_rss_delta_mb": session_rss_delta_mb,
    "first_inference_ms": first_inference_ms,
    **stats,
    "final_rss_mb": final_rss_mb,
    "peak_rss_mb": peak_memory_mb,
}

write_json(RAW_PATH, raw_result)
write_json(METRICS_PATH, metrics)
write_json(ENVIRONMENT_PATH, collect_environment(ROOT_DIR))

print(f"Provider: {session.get_providers()}")
print(f"ONNX Runtime threads: {args.threads}")
print(f"Batch size: {args.batch}")
print(f"First inference: {first_inference_ms:.4f} ms")
print(f"Final RSS: {final_rss_mb:.2f} MB")
print(f"Peak RSS: {peak_memory_mb:.2f} MB")
print(f"Output saved: {OUTPUT_PATH}")
print(f"Raw latency saved: {RAW_PATH}")
print(f"Metrics saved: {METRICS_PATH}")
