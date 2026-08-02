import argparse
import json
import resource
import socket
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
import psutil

from benchmark import benchmark, print_statistics


def current_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def summarize(times: list[float], batch_size: int) -> dict:
    mean_ms = float(np.mean(times))

    return {
        "mean_batch_latency_ms": mean_ms,
        "median_batch_latency_ms": float(np.median(times)),
        "p95_batch_latency_ms": float(np.percentile(times, 95)),
        "min_batch_latency_ms": float(min(times)),
        "max_batch_latency_ms": float(max(times)),
        "per_image_latency_ms": mean_ms / batch_size,
        "throughput_images_per_second": (
            batch_size / (mean_ms / 1000)
        ),
    }


parser = argparse.ArgumentParser(
    description="ONNX Runtime CPU/CUDA benchmark"
)

parser.add_argument(
    "--provider",
    choices=["cpu", "cuda"],
    default="cpu",
)

parser.add_argument(
    "--threads",
    type=int,
    default=4,
)

parser.add_argument(
    "--batch",
    type=int,
    required=True,
    choices=[1, 4, 16],
)

parser.add_argument(
    "--warmup",
    type=int,
    default=10,
)

parser.add_argument(
    "--measurements",
    type=int,
    default=300,
)

args = parser.parse_args()


BENCHMARK_DIR = Path(__file__).resolve().parent

INPUT_DIR = BENCHMARK_DIR / "data" / "inputs"
MODEL_DIR = BENCHMARK_DIR / "data" / "onnx_models"
RESULT_DIR = BENCHMARK_DIR / "results" / "onnx"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = INPUT_DIR / f"input_b{args.batch}.npy"
MODEL_PATH = MODEL_DIR / f"resnet18_b{args.batch}.onnx"

if args.provider == "cpu":
    OUTPUT_PATH = (
        RESULT_DIR / f"onnx_output_b{args.batch}.npy"
    )
    METRICS_PATH = (
        RESULT_DIR
        / f"onnx_metrics_b{args.batch}_t{args.threads}.json"
    )
else:
    OUTPUT_PATH = (
        RESULT_DIR / f"onnx_output_cuda_b{args.batch}.npy"
    )
    METRICS_PATH = (
        RESULT_DIR / f"onnx_metrics_cuda_b{args.batch}.json"
    )


if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input not found: {INPUT_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}"
    )


start = time.perf_counter()
input_array = np.load(INPUT_PATH)
input_load_ms = (time.perf_counter() - start) * 1000

if input_array.ndim != 4:
    raise ValueError(
        f"Expected 4D input, got {input_array.shape}"
    )

if input_array.shape[0] != args.batch:
    raise ValueError(
        f"Batch mismatch: expected {args.batch}, "
        f"got {input_array.shape[0]}"
    )


options = ort.SessionOptions()
options.intra_op_num_threads = args.threads
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL


if args.provider == "cuda":
    ort.preload_dlls(directory="")
    available_providers = ort.get_available_providers()
    
    if "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. "
            f"Available: {available_providers}"
        )

    requested_providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
else:
    requested_providers = ["CPUExecutionProvider"]


rss_before_session = current_rss_mb()

start = time.perf_counter()

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=requested_providers,
)

session_load_ms = (
    time.perf_counter() - start
) * 1000

rss_after_session = current_rss_mb()


actual_providers = session.get_providers()

if (
    args.provider == "cuda"
    and "CUDAExecutionProvider" not in actual_providers
):
    raise RuntimeError(
        f"CUDA provider was not loaded: {actual_providers}"
    )


model_input = session.get_inputs()[0]
model_output = session.get_outputs()[0]

input_name = model_input.name
output_name = model_output.name


# 기본 session.run은 NumPy CPU 입력과 CPU 출력을 사용한다.
start = time.perf_counter()

first_outputs = session.run(
    None,
    {input_name: input_array},
)

first_inference_ms = (
    time.perf_counter() - start
) * 1000

first_output = first_outputs[0]

np.save(OUTPUT_PATH, first_output)


print("Hostname:", socket.gethostname())
print("Provider request:", args.provider)
print("Actual providers:", actual_providers)
print("Batch:", args.batch)
print("Input shape:", input_array.shape)
print("Output shape:", first_output.shape)
print(f"Input loading: {input_load_ms:.4f} ms")
print(f"Session loading: {session_load_ms:.4f} ms")
print(f"First inference: {first_inference_ms:.4f} ms")
print("Output saved:", OUTPUT_PATH)


metrics = {
    "runtime": "onnxruntime",
    "onnxruntime_version": ort.__version__,
    "hostname": socket.gethostname(),
    "provider_request": args.provider,
    "session_providers": actual_providers,
    "model": "resnet18",
    "model_path": str(MODEL_PATH),
    "input_path": str(INPUT_PATH),
    "output_path": str(OUTPUT_PATH),
    "batch_size": args.batch,
    "input_shape": list(input_array.shape),
    "output_shape": list(first_output.shape),
    "input_dtype": str(input_array.dtype),
    "warmup_count": args.warmup,
    "measurement_count": args.measurements,
    "input_load_ms": input_load_ms,
    "session_load_ms": session_load_ms,
    "first_inference_ms": first_inference_ms,
    "rss_before_session_mb": rss_before_session,
    "rss_after_session_mb": rss_after_session,
    "session_rss_delta_mb": (
        rss_after_session - rss_before_session
    ),
    "intra_op_threads": args.threads,
    "inter_op_threads": 1,
}


# --------------------------------------------------
# CPU
# --------------------------------------------------

if args.provider == "cpu":

    def cpu_inference():
        return session.run(
            None,
            {input_name: input_array},
        )

    times = benchmark(
        inference_fn=cpu_inference,
        warming_count=args.warmup,
        measured_count=args.measurements,
    )

    print("\n=== ONNX Runtime CPU ===")
    print_statistics(times)

    metrics["cpu_inference"] = summarize(
        times,
        args.batch,
    )


# --------------------------------------------------
# CUDA
# --------------------------------------------------

else:
    # NumPy CPU 입력 → CUDA → CPU 출력
    def end_to_end_inference():
        return session.run(
            None,
            {input_name: input_array},
        )

    end_to_end_times = benchmark(
        inference_fn=end_to_end_inference,
        warming_count=args.warmup,
        measured_count=args.measurements,
    )

    print("\n=== ONNX CUDA End-to-End ===")
    print_statistics(end_to_end_times)

    end_to_end_result = summarize(
        end_to_end_times,
        args.batch,
    )

    # 입력과 출력을 GPU에 고정한다.
    device_input = ort.OrtValue.ortvalue_from_numpy(
        input_array,
        "cuda",
        0,
    )

    device_output = (
        ort.OrtValue.ortvalue_from_shape_and_type(
            first_output.shape,
            first_output.dtype,
            "cuda",
            0,
        )
    )

    io_binding = session.io_binding()

    io_binding.bind_ortvalue_input(
        input_name,
        device_input,
    )

    io_binding.bind_ortvalue_output(
        output_name,
        device_output,
    )

    def gpu_only_inference():
        session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()

    gpu_only_times = benchmark(
        inference_fn=gpu_only_inference,
        warming_count=args.warmup,
        measured_count=args.measurements,
    )

    print("\n=== ONNX CUDA GPU-only IOBinding ===")
    print_statistics(gpu_only_times)

    gpu_only_result = summarize(
        gpu_only_times,
        args.batch,
    )

    metrics.update({
        "gpu_only_method": "IOBinding",
        "gpu_only": gpu_only_result,
        "end_to_end": end_to_end_result,
    })


metrics["final_rss_mb"] = current_rss_mb()
metrics["peak_rss_mb"] = peak_rss_mb()


with METRICS_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metrics,
        file,
        indent=2,
        ensure_ascii=False,
    )

print("\nMetrics saved:", METRICS_PATH)
