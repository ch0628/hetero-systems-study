from __future__ import annotations

import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil


def benchmark(
    inference_fn: Callable[[], Any],
    warming_count: int = 10,
    measured_count: int = 300,
) -> list[float]:
    if warming_count < 0:
        raise ValueError("warming_count must be >= 0")
    if measured_count <= 0:
        raise ValueError("measured_count must be > 0")

    for _ in range(warming_count):
        inference_fn()

    times_ms: list[float] = []
    for _ in range(measured_count):
        start_time = time.perf_counter()
        inference_fn()
        end_time = time.perf_counter()
        times_ms.append((end_time - start_time) * 1000.0)

    return times_ms


def calculate_statistics(times_ms: list[float], batch_size: int) -> dict[str, float | int]:
    if not times_ms:
        raise ValueError("times_ms is empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    values = np.asarray(times_ms, dtype=np.float64)
    count = int(values.size)
    mean_ms = float(np.mean(values))
    std_dev_ms = float(np.std(values, ddof=1)) if count >= 2 else 0.0
    ci95_half_width_ms = 1.96 * std_dev_ms / math.sqrt(count)

    return {
        "measurement_count": count,
        "mean_batch_latency_ms": mean_ms,
        "median_batch_latency_ms": float(np.median(values)),
        "p95_batch_latency_ms": float(np.percentile(values, 95)),
        "p99_batch_latency_ms": float(np.percentile(values, 99)),
        "min_batch_latency_ms": float(np.min(values)),
        "max_batch_latency_ms": float(np.max(values)),
        "std_dev_batch_latency_ms": std_dev_ms,
        "ci95_lower_batch_latency_ms": mean_ms - ci95_half_width_ms,
        "ci95_upper_batch_latency_ms": mean_ms + ci95_half_width_ms,
        "ci95_half_width_batch_latency_ms": ci95_half_width_ms,
        "per_image_latency_ms": mean_ms / batch_size,
        "throughput_images_per_second": batch_size * 1000.0 / mean_ms,
    }


def print_statistics(stats: dict[str, float | int]) -> None:
    print(f"Mean: {float(stats['mean_batch_latency_ms']):.4f} ms")
    print(f"Median: {float(stats['median_batch_latency_ms']):.4f} ms")
    print(f"P95: {float(stats['p95_batch_latency_ms']):.4f} ms")
    print(f"P99: {float(stats['p99_batch_latency_ms']):.4f} ms")
    print(f"Min: {float(stats['min_batch_latency_ms']):.4f} ms")
    print(f"Max: {float(stats['max_batch_latency_ms']):.4f} ms")
    print(f"Std Dev: {float(stats['std_dev_batch_latency_ms']):.4f} ms")
    print(
        "95% Confidence Interval: "
        f"[{float(stats['ci95_lower_batch_latency_ms']):.4f}, "
        f"{float(stats['ci95_upper_batch_latency_ms']):.4f}] ms"
    )
    print(f"Per-image Latency: {float(stats['per_image_latency_ms']):.4f} ms")
    print(
        "Throughput: "
        f"{float(stats['throughput_images_per_second']):.4f} images/sec"
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str:
    processor = platform.processor().strip()
    if processor:
        return processor

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return "unknown"


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_environment(repo_root: Path) -> dict[str, Any]:
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "cpu_model": _cpu_model(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 3),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "virtual_environment": os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX"),
        "package_versions": {
            "numpy": _package_version("numpy"),
            "psutil": _package_version("psutil"),
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "onnx": _package_version("onnx"),
            "onnxruntime": _package_version("onnxruntime"),
        },
        "git_commit": _git_commit(repo_root),
    }
