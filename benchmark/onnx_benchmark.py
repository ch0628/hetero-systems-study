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
import onnxruntime as ort
import psutil

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
        description="ONNX Runtime CPU/CUDA benchmark with raw repeated-run results"
    )
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
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
        help="Explicitly control CUDAExecutionProvider TF32.",
    )
    parser.add_argument("--enable-profiler", action="store_true")
    parser.add_argument(
        "--disable-cpu-fallback",
        action="store_true",
        help="Request CUDAExecutionProvider only. Unsupported nodes will fail instead of falling back.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be > 0")
    if args.warmup < 0 or args.measurements <= 0:
        raise ValueError("warmup must be >= 0 and measurements must be > 0")

    run_id = safe_token(args.run_id)
    np.random.seed(args.seed)

    benchmark_dir = Path(__file__).resolve().parent
    input_path = benchmark_dir / "data" / "inputs" / f"input_b{args.batch}.npy"
    model_path = benchmark_dir / "data" / "onnx_models" / f"resnet18_b{args.batch}.onnx"
    result_dir = benchmark_dir / "results" / "onnx"
    raw_dir = benchmark_dir / "results" / "raw"
    profile_dir = benchmark_dir / "results" / "profiles" / "onnxruntime"
    result_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.provider}_b{args.batch}_{run_id}"
    if args.provider == "cpu":
        suffix = f"cpu_b{args.batch}_t{args.threads}_{run_id}"

    output_path = result_dir / f"onnx_output_{suffix}.npy"
    canonical_output_path = result_dir / f"onnx_output_{args.provider}_b{args.batch}.npy"
    metrics_path = result_dir / f"onnx_metrics_{suffix}.json"
    raw_path = raw_dir / f"onnx_raw_{suffix}.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    started_at = datetime.now(timezone.utc).isoformat()
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

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_profiling = args.enable_profiler

    if args.provider == "cuda":
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. "
                f"Available providers: {available_providers}"
            )
        cuda_provider_options = {
            "device_id": 0,
            "use_tf32": 1 if args.tf32 == "on" else 0,
        }
        requested_providers: list[Any] = [
            ("CUDAExecutionProvider", cuda_provider_options)
        ]
        if not args.disable_cpu_fallback:
            requested_providers.append("CPUExecutionProvider")
    else:
        available_providers = ort.get_available_providers()
        requested_providers = ["CPUExecutionProvider"]

    rss_before_session = current_rss_mb()
    start = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=requested_providers,
    )
    session_load_ms = (time.perf_counter() - start) * 1000.0
    rss_after_session = current_rss_mb()

    actual_providers = session.get_providers()
    provider_options = session.get_provider_options()
    if args.provider == "cuda" and "CUDAExecutionProvider" not in actual_providers:
        raise RuntimeError(f"CUDA provider was not loaded: {actual_providers}")

    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    input_name = model_input.name
    output_name = model_output.name

    start = time.perf_counter()
    first_outputs = session.run(None, {input_name: input_array})
    first_inference_ms = (time.perf_counter() - start) * 1000.0
    first_output = first_outputs[0]
    np.save(output_path, first_output)
    shutil.copyfile(output_path, canonical_output_path)

    raw_sections: dict[str, list[float]] = {}
    section_summaries: dict[str, dict[str, Any]] = {}
    io_binding_prepare_ms = 0.0
    gpu_only_method: str | None = None

    if args.provider == "cpu":
        def cpu_inference() -> list[np.ndarray]:
            return session.run(None, {input_name: input_array})

        cpu_times = benchmark(cpu_inference, args.warmup, args.measurements)
        raw_sections["cpu_inference"] = cpu_times
        section_summaries["cpu_inference"] = summarize_latency(cpu_times, args.batch)
        print("\n=== ONNX Runtime CPU ===")
        print_statistics(cpu_times)
    else:
        def end_to_end_inference() -> list[np.ndarray]:
            return session.run(None, {input_name: input_array})

        end_to_end_times = benchmark(
            end_to_end_inference,
            args.warmup,
            args.measurements,
        )
        raw_sections["end_to_end"] = end_to_end_times
        section_summaries["end_to_end"] = summarize_latency(
            end_to_end_times,
            args.batch,
        )
        print("\n=== ONNX CUDA End-to-End ===")
        print_statistics(end_to_end_times)

        start = time.perf_counter()
        device_input = ort.OrtValue.ortvalue_from_numpy(input_array, "cuda", 0)
        device_output = ort.OrtValue.ortvalue_from_shape_and_type(
            first_output.shape,
            first_output.dtype,
            "cuda",
            0,
        )
        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input(input_name, device_input)
        io_binding.bind_ortvalue_output(output_name, device_output)
        io_binding.synchronize_inputs()
        io_binding_prepare_ms = (time.perf_counter() - start) * 1000.0

        def gpu_only_inference() -> None:
            session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

        gpu_only_times = benchmark(
            gpu_only_inference,
            args.warmup,
            args.measurements,
        )
        raw_sections["gpu_only"] = gpu_only_times
        section_summaries["gpu_only"] = summarize_latency(
            gpu_only_times,
            args.batch,
        )
        gpu_only_method = "IOBinding with GPU-resident input/output"
        print("\n=== ONNX CUDA GPU-only IOBinding ===")
        print_statistics(gpu_only_times)

    profile_path: str | None = None
    if args.enable_profiler:
        generated_profile = Path(session.end_profiling())
        destination = profile_dir / f"onnx_profile_{args.provider}_b{args.batch}_{run_id}.json"
        if generated_profile.exists():
            shutil.move(str(generated_profile), destination)
            profile_path = str(destination)

    common_metadata = {
        "schema_version": 2,
        "runtime": "onnxruntime",
        "provider_request": args.provider,
        "run_id": run_id,
        "run_order": args.run_order,
        "started_at_utc": started_at,
        "hostname": socket.gethostname(),
        "seed": args.seed,
        "model": "resnet18",
        "precision": "fp32",
        "tf32": args.tf32 if args.provider == "cuda" else "not_applicable",
        "batch_size": args.batch,
        "warmup_count": args.warmup,
        "measurement_count": args.measurements,
    }
    save_raw_latencies(raw_path, metadata=common_metadata, sections=raw_sections)

    metrics: dict[str, Any] = {
        **common_metadata,
        "raw_latency_path": str(raw_path),
        "model_path": str(model_path),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "canonical_output_path": str(canonical_output_path),
        "input_shape": list(input_array.shape),
        "output_shape": list(first_output.shape),
        "input_dtype": str(input_array.dtype),
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "onnxruntime_version": ort.__version__,
        "available_providers": available_providers,
        "requested_providers": requested_providers,
        "session_providers": actual_providers,
        "provider_options": provider_options,
        "cpu_fallback_allowed": not args.disable_cpu_fallback,
        "nvidia_driver_version": nvidia_value("driver_version") if args.provider == "cuda" else None,
        "gpu_name": nvidia_value("name") if args.provider == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "input_load_ms": input_load_ms,
        "session_load_ms": session_load_ms,
        "first_inference_ms": first_inference_ms,
        "io_binding_prepare_ms": io_binding_prepare_ms,
        "rss_before_session_mb": rss_before_session,
        "rss_after_session_mb": rss_after_session,
        "session_rss_delta_mb": rss_after_session - rss_before_session,
        "final_rss_mb": current_rss_mb(),
        "peak_rss_mb": peak_rss_mb(),
        "intra_op_threads": args.threads,
        "inter_op_threads": 1,
        "execution_mode": "ORT_SEQUENTIAL",
        "sections": section_summaries,
        "gpu_only_method": gpu_only_method,
        "profiler_trace_path": profile_path,
        "measurement_boundary": {
            "gpu_only": "IOBinding keeps input and output on GPU; includes ORT launch and output synchronization",
            "end_to_end": "NumPy CPU input, ORT CUDA execution, and NumPy CPU output",
        } if args.provider == "cuda" else {
            "cpu_inference": "session and NumPy input already resident in the process"
        },
        "tf32_control": "explicit CUDAExecutionProvider use_tf32 option",
    }
    save_json(metrics_path, metrics)

    print("Hostname:", socket.gethostname())
    print("Provider request:", args.provider)
    print("Requested providers:", requested_providers)
    print("Actual providers:", actual_providers)
    print("Provider options:", provider_options)
    print("Run ID:", run_id)
    print("Run order:", args.run_order)
    print("Model: resnet18")
    print("Precision: fp32")
    print("TF32:", metrics["tf32"])
    print("ONNX Runtime:", ort.__version__)
    if args.provider == "cuda":
        print("GPU:", metrics["gpu_name"])
        print("NVIDIA driver:", metrics["nvidia_driver_version"])
    print("Batch:", args.batch)
    print("Warm-up:", args.warmup)
    print("Measurements:", args.measurements)
    print(f"Session loading: {session_load_ms:.4f} ms")
    print(f"First inference: {first_inference_ms:.4f} ms")
    print("Raw latency saved:", raw_path)
    print("Metrics saved:", metrics_path)
    print("Output saved:", output_path)
    if profile_path:
        print("Profiler trace saved:", profile_path)


if __name__ == "__main__":
    main()
