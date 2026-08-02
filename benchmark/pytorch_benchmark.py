import argparse
import json
import resource
import socket
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torchvision.models import resnet18, ResNet18_Weights

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
    description="PyTorch CPU/CUDA inference benchmark"
)

parser.add_argument(
    "--device",
    choices=["cpu", "cuda"],
    default="cpu",
)

parser.add_argument(
    "--threads",
    type=int,
    default=4,
    help="CPU threads. Used only when device=cpu.",
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


if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device = torch.device(args.device)


BENCHMARK_DIR = Path(__file__).resolve().parent
INPUT_DIR = BENCHMARK_DIR / "data" / "inputs"
RESULT_DIR = BENCHMARK_DIR / "results" / "pytorch"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = INPUT_DIR / f"input_b{args.batch}.npy"

if args.device == "cpu":
    OUTPUT_PATH = (
        RESULT_DIR / f"pytorch_output_b{args.batch}.npy"
    )
    METRICS_PATH = (
        RESULT_DIR
        / f"pytorch_metrics_b{args.batch}_t{args.threads}.json"
    )
else:
    OUTPUT_PATH = (
        RESULT_DIR / f"pytorch_output_cuda_b{args.batch}.npy"
    )
    METRICS_PATH = (
        RESULT_DIR / f"pytorch_metrics_cuda_b{args.batch}.json"
    )


if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input not found: {INPUT_PATH}\n"
        f"Run: python benchmark/export_onnx.py "
        f"--batch {args.batch}"
    )


if args.device == "cpu":
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)


# --------------------------------------------------
# 모델 초기화
# --------------------------------------------------

rss_before_model = current_rss_mb()

start = time.perf_counter()

model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)
model.eval()

model_initialization_ms = (
    time.perf_counter() - start
) * 1000

rss_after_model = current_rss_mb()


# --------------------------------------------------
# 모델을 실행 장치로 전송
# --------------------------------------------------

model_transfer_ms = 0.0

if device.type == "cuda":
    torch.cuda.synchronize()

    start = time.perf_counter()
    model = model.to(device)
    torch.cuda.synchronize()

    model_transfer_ms = (
        time.perf_counter() - start
    ) * 1000


# --------------------------------------------------
# 입력 로딩 및 Tensor 변환
# --------------------------------------------------

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

start = time.perf_counter()
input_cpu = torch.from_numpy(input_array).contiguous()
tensor_conversion_ms = (
    time.perf_counter() - start
) * 1000


# --------------------------------------------------
# GPU-only 측정을 위한 입력 전송
# --------------------------------------------------

input_transfer_ms = 0.0

if device.type == "cuda":
    torch.cuda.synchronize()

    start = time.perf_counter()
    input_device = input_cpu.to(device)
    torch.cuda.synchronize()

    input_transfer_ms = (
        time.perf_counter() - start
    ) * 1000
else:
    input_device = input_cpu


# --------------------------------------------------
# 첫 추론
# --------------------------------------------------

if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

with torch.inference_mode():
    first_output = model(input_device)

if device.type == "cuda":
    torch.cuda.synchronize()

first_inference_ms = (
    time.perf_counter() - start
) * 1000


output_array = (
    first_output
    .detach()
    .cpu()
    .numpy()
)

np.save(OUTPUT_PATH, output_array)


print("Hostname:", socket.gethostname())
print("Device:", device)
print("Batch:", args.batch)
print("Input shape:", input_array.shape)
print("Output shape:", output_array.shape)
print(f"Model initialization: {model_initialization_ms:.4f} ms")
print(f"Model transfer: {model_transfer_ms:.4f} ms")
print(f"Input loading: {input_load_ms:.4f} ms")
print(f"Tensor conversion: {tensor_conversion_ms:.4f} ms")
print(f"Input transfer: {input_transfer_ms:.4f} ms")
print(f"First inference: {first_inference_ms:.4f} ms")
print("Output saved:", OUTPUT_PATH)


metrics = {
    "runtime": "pytorch",
    "hostname": socket.gethostname(),
    "device": args.device,
    "model": "resnet18",
    "batch_size": args.batch,
    "input_path": str(INPUT_PATH),
    "output_path": str(OUTPUT_PATH),
    "input_shape": list(input_array.shape),
    "output_shape": list(output_array.shape),
    "input_dtype": str(input_array.dtype),
    "warmup_count": args.warmup,
    "measurement_count": args.measurements,
    "input_load_ms": input_load_ms,
    "tensor_conversion_ms": tensor_conversion_ms,
    "model_initialization_ms": model_initialization_ms,
    "model_transfer_ms": model_transfer_ms,
    "input_transfer_ms": input_transfer_ms,
    "first_inference_ms": first_inference_ms,
    "rss_before_model_mb": rss_before_model,
    "rss_after_model_mb": rss_after_model,
}


# --------------------------------------------------
# CPU Benchmark
# --------------------------------------------------

if device.type == "cpu":

    def cpu_inference():
        with torch.inference_mode():
            return model(input_cpu)

    times = benchmark(
        inference_fn=cpu_inference,
        warming_count=args.warmup,
        measured_count=args.measurements,
    )

    print("\n=== CPU inference ===")
    print_statistics(times)

    cpu_result = summarize(times, args.batch)

    metrics.update({
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "cpu_inference": cpu_result,
    })


# --------------------------------------------------
# CUDA Benchmark
# --------------------------------------------------

else:
    gpu_properties = torch.cuda.get_device_properties(0)

    print("\nGPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())

    # 입력과 모델이 이미 GPU에 있는 상태
    def gpu_only_inference():
        with torch.inference_mode():
            output = model(input_device)

        torch.cuda.synchronize()
        return output

    torch.cuda.reset_peak_memory_stats()

    gpu_only_times = benchmark(
        inference_fn=gpu_only_inference,
        warming_count=args.warmup,
        measured_count=args.measurements,
    )

    gpu_only_peak_allocated_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
    )
    gpu_only_peak_reserved_mb = (
        torch.cuda.max_memory_reserved() / (1024 ** 2)
    )

    print("\n=== CUDA GPU-only inference ===")
    print_statistics(gpu_only_times)

    gpu_only_result = summarize(
        gpu_only_times,
        args.batch,
    )

    # CPU 입력 → GPU → 추론 → CPU 결과
    def end_to_end_inference():
        with torch.inference_mode():
            current_input = input_cpu.to(device)
            output = model(current_input)
            output_cpu = output.cpu()

        torch.cuda.synchronize()
        return output_cpu

    torch.cuda.reset_peak_memory_stats()

    end_to_end_times = benchmark(
        inference_fn=end_to_end_inference,
        warming_count=args.warmup,
        measured_count=args.measurements,
    )

    end_to_end_peak_allocated_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
    )
    end_to_end_peak_reserved_mb = (
        torch.cuda.max_memory_reserved() / (1024 ** 2)
    )

    print("\n=== CUDA End-to-End inference ===")
    print_statistics(end_to_end_times)

    end_to_end_result = summarize(
        end_to_end_times,
        args.batch,
    )

    metrics.update({
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_mb": (
            gpu_properties.total_memory / (1024 ** 2)
        ),
        "host_memory_pinned": False,
        "gpu_only": {
            **gpu_only_result,
            "peak_allocated_mb": (
                gpu_only_peak_allocated_mb
            ),
            "peak_reserved_mb": (
                gpu_only_peak_reserved_mb
            ),
        },
        "end_to_end": {
            **end_to_end_result,
            "peak_allocated_mb": (
                end_to_end_peak_allocated_mb
            ),
            "peak_reserved_mb": (
                end_to_end_peak_reserved_mb
            ),
        },
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
