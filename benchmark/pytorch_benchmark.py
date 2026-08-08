from __future__ import annotations

import argparse
import os
import platform
import resource
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torchvision
from torchvision.models import ResNet18_Weights, resnet18

from benchmark import (
    benchmark,
    print_statistics,
    safe_token,
    save_json,
    save_raw_latencies,
    summarize_latency,
)


def current_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024 if sys.platform != "darwin" else value / (1024**2)


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = completed.stdout.strip()
    return output or None


def nvidia_value(field: str) -> str | None:
    output = command_output(
        [
            "nvidia-smi",
            f"--query-gpu={field}",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ]
    )
    return output.splitlines()[0].strip() if output else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch CPU/CUDA inference benchmark with repeatable raw results"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, required=True, choices=[1, 4, 16])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measurements", type=int, default=300)
    parser.add_argument("--run-id", default="single")
    parser.add_argument("--run-order", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tf32",
        choices=["on", "off"],
        default="off",
        help="Explicitly control PyTorch CUDA TF32 for validation reproducibility.",
    )
    parser.add_argument("--enable-profiler", action="store_true")
    parser.add_argument("--profile-iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be > 0")
    if args.warmup < 0 or args.measurements <= 0:
        raise ValueError("warmup must be >= 0 and measurements must be > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch environment.")

    run_id = safe_token(args.run_id)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    else:
        allow_tf32 = args.tf32 == "on"
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = False

    benchmark_dir = Path(__file__).resolve().parent
    input_path = benchmark_dir / "data" / "inputs" / f"input_b{args.batch}.npy"
    result_dir = benchmark_dir / "results" / "pytorch"
    raw_dir = benchmark_dir / "results" / "raw"
    profile_dir = benchmark_dir / "results" / "profiles" / "pytorch"
    result_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.device}_b{args.batch}_{run_id}"
    if device.type == "cpu":
        suffix = f"cpu_b{args.batch}_t{args.threads}_{run_id}"

    output_path = result_dir / f"pytorch_output_{suffix}.npy"
    canonical_output_path = result_dir / f"pytorch_output_{args.device}_b{args.batch}.npy"
    metrics_path = result_dir / f"pytorch_metrics_{suffix}.json"
    raw_path = raw_dir / f"pytorch_raw_{suffix}.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {input_path}\n"
            f"Run: python benchmark/export_onnx.py --batch {args.batch}"
        )

    started_at = datetime.now(timezone.utc).isoformat()
    rss_before_model = current_rss_mb()
    start = time.perf_counter()
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()
    model_initialization_ms = (time.perf_counter() - start) * 1000.0
    rss_after_model = current_rss_mb()

    cuda_context_init_ms = 0.0
    model_transfer_ms = 0.0
    if device.type == "cuda":
        start = time.perf_counter()
        torch.cuda.init()
        torch.cuda.synchronize()
        cuda_context_init_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        model = model.to(device)
        torch.cuda.synchronize()
        model_transfer_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    input_array = np.load(input_path)
    input_load_ms = (time.perf_counter() - start) * 1000.0
    if input_array.ndim != 4 or input_array.shape[0] != args.batch:
        raise ValueError(
            f"Expected input shape (batch, 3, 224, 224) with batch={args.batch}, "
            f"got {input_array.shape}"
        )
    if input_array.dtype != np.float32:
        raise ValueError(f"Expected FP32 input, got {input_array.dtype}")

    start = time.perf_counter()
    input_cpu = torch.from_numpy(input_array).contiguous()
    tensor_conversion_ms = (time.perf_counter() - start) * 1000.0

    input_transfer_ms = 0.0
    if device.type == "cuda":
        start = time.perf_counter()
        input_device = input_cpu.to(device)
        torch.cuda.synchronize()
        input_transfer_ms = (time.perf_counter() - start) * 1000.0
    else:
        input_device = input_cpu

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        first_output = model(input_device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    first_inference_ms = (time.perf_counter() - start) * 1000.0

    output_array = first_output.detach().cpu().numpy()
    np.save(output_path, output_array)
    shutil.copyfile(output_path, canonical_output_path)

    raw_sections: dict[str, list[float]] = {}
    section_summaries: dict[str, dict[str, Any]] = {}
    gpu_memory: dict[str, Any] = {}

    if device.type == "cpu":
        def cpu_inference() -> torch.Tensor:
            with torch.inference_mode():
                return model(input_cpu)

        cpu_times = benchmark(cpu_inference, args.warmup, args.measurements)
        raw_sections["cpu_inference"] = cpu_times
        section_summaries["cpu_inference"] = summarize_latency(cpu_times, args.batch)
        print("\n=== CPU inference ===")
        print_statistics(cpu_times)
    else:
        model_allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        model_reserved_mb = torch.cuda.memory_reserved() / (1024**2)

        def gpu_only_inference() -> torch.Tensor:
            with torch.inference_mode():
                output = model(input_device)
            torch.cuda.synchronize()
            return output

        torch.cuda.reset_peak_memory_stats()
        gpu_only_times = benchmark(gpu_only_inference, args.warmup, args.measurements)
        raw_sections["gpu_only"] = gpu_only_times
        section_summaries["gpu_only"] = summarize_latency(gpu_only_times, args.batch)
        gpu_memory["gpu_only_peak_allocated_mb"] = (
            torch.cuda.max_memory_allocated() / (1024**2)
        )
        gpu_memory["gpu_only_peak_reserved_mb"] = (
            torch.cuda.max_memory_reserved() / (1024**2)
        )
        print("\n=== CUDA GPU-only inference ===")
        print_statistics(gpu_only_times)

        def end_to_end_inference() -> torch.Tensor:
            with torch.inference_mode():
                current_input = input_cpu.to(device)
                output = model(current_input)
                output_cpu = output.cpu()
            torch.cuda.synchronize()
            return output_cpu

        torch.cuda.reset_peak_memory_stats()
        end_to_end_times = benchmark(end_to_end_inference, args.warmup, args.measurements)
        raw_sections["end_to_end"] = end_to_end_times
        section_summaries["end_to_end"] = summarize_latency(end_to_end_times, args.batch)
        gpu_memory["end_to_end_peak_allocated_mb"] = (
            torch.cuda.max_memory_allocated() / (1024**2)
        )
        gpu_memory["end_to_end_peak_reserved_mb"] = (
            torch.cuda.max_memory_reserved() / (1024**2)
        )
        gpu_memory["model_loaded_allocated_mb"] = model_allocated_mb
        gpu_memory["model_loaded_reserved_mb"] = model_reserved_mb
        print("\n=== CUDA End-to-End inference ===")
        print_statistics(end_to_end_times)

    profile_path: str | None = None
    if args.enable_profiler:
        if device.type != "cuda":
            raise ValueError("--enable-profiler is supported only with --device cuda")
        from torch.profiler import ProfilerActivity, profile

        trace_path = profile_dir / f"pytorch_profile_cuda_b{args.batch}_{run_id}.json"
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            for _ in range(args.profile_iterations):
                with torch.inference_mode():
                    model(input_device)
                torch.cuda.synchronize()
        profiler.export_chrome_trace(str(trace_path))
        profile_path = str(trace_path)

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    gpu_total_memory_mb = None
    if device.type == "cuda":
        gpu_total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)

    common_metadata = {
        "schema_version": 2,
        "runtime": "pytorch",
        "device": args.device,
        "run_id": run_id,
        "run_order": args.run_order,
        "started_at_utc": started_at,
        "hostname": socket.gethostname(),
        "seed": args.seed,
        "model": "resnet18",
        "weights": "ResNet18_Weights.DEFAULT",
        "precision": "fp32",
        "tf32": args.tf32 if device.type == "cuda" else "not_applicable",
        "batch_size": args.batch,
        "warmup_count": args.warmup,
        "measurement_count": args.measurements,
    }
    save_raw_latencies(raw_path, metadata=common_metadata, sections=raw_sections)

    metrics: dict[str, Any] = {
        **common_metadata,
        "raw_latency_path": str(raw_path),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "canonical_output_path": str(canonical_output_path),
        "input_shape": list(input_array.shape),
        "output_shape": list(output_array.shape),
        "input_dtype": str(input_array.dtype),
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver_version": nvidia_value("driver_version") if device.type == "cuda" else None,
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_memory_mb,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "input_load_ms": input_load_ms,
        "tensor_conversion_ms": tensor_conversion_ms,
        "model_initialization_ms": model_initialization_ms,
        "cuda_context_init_ms": cuda_context_init_ms,
        "model_transfer_ms": model_transfer_ms,
        "input_transfer_ms": input_transfer_ms,
        "first_inference_ms": first_inference_ms,
        "rss_before_model_mb": rss_before_model,
        "rss_after_model_mb": rss_after_model,
        "model_rss_delta_mb": rss_after_model - rss_before_model,
        "final_rss_mb": current_rss_mb(),
        "peak_rss_mb": peak_rss_mb(),
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "sections": section_summaries,
        "gpu_memory": gpu_memory,
        "profiler_trace_path": profile_path,
        "measurement_boundary": {
            "gpu_only": "model and input already on GPU; includes runtime launch and per-iteration synchronization",
            "end_to_end": "CPU tensor to GPU, inference, GPU output to CPU, and synchronization",
        } if device.type == "cuda" else {
            "cpu_inference": "model and input tensor already resident in the process"
        },
    }
    save_json(metrics_path, metrics)

    print("Hostname:", socket.gethostname())
    print("Device:", device)
    print("Run ID:", run_id)
    print("Run order:", args.run_order)
    print("Model: resnet18")
    print("Precision: fp32")
    print("TF32:", metrics["tf32"])
    if device.type == "cuda":
        print("GPU:", gpu_name)
        print("PyTorch:", torch.__version__)
        print("CUDA runtime:", torch.version.cuda)
        print("cuDNN:", torch.backends.cudnn.version())
        print("NVIDIA driver:", metrics["nvidia_driver_version"])
    print("Batch:", args.batch)
    print("Warm-up:", args.warmup)
    print("Measurements:", args.measurements)
    print(f"First inference: {first_inference_ms:.4f} ms")
    print("Raw latency saved:", raw_path)
    print("Metrics saved:", metrics_path)
    print("Output saved:", output_path)
    if profile_path:
        print("Profiler trace saved:", profile_path)


if __name__ == "__main__":
    main()
