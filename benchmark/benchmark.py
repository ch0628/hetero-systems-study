from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


LatencyFn = Callable[[], Any]


def benchmark(
    inference_fn: LatencyFn,
    warming_count: int = 10,
    measured_count: int = 300,
) -> list[float]:
    """Run warm-up calls, then return one wall-clock latency per measured call."""
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


def summarize_latency(times_ms: list[float], batch_size: int) -> dict[str, Any]:
    """Calculate the common latency and throughput statistics used by both runtimes."""
    if not times_ms:
        raise ValueError("times_ms must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    count = len(times_ms)
    mean_ms = float(statistics.fmean(times_ms))
    stdev_ms = float(statistics.stdev(times_ms)) if count > 1 else 0.0
    standard_error_ms = stdev_ms / math.sqrt(count) if count > 1 else 0.0
    ci_margin_ms = 1.96 * standard_error_ms

    return {
        "measurement_count": count,
        "mean_batch_latency_ms": mean_ms,
        "median_batch_latency_ms": float(statistics.median(times_ms)),
        "p95_batch_latency_ms": float(np.percentile(times_ms, 95)),
        "p99_batch_latency_ms": float(np.percentile(times_ms, 99)),
        "min_batch_latency_ms": float(min(times_ms)),
        "max_batch_latency_ms": float(max(times_ms)),
        "sample_stdev_ms": stdev_ms,
        "mean_ci95_lower_ms": mean_ms - ci_margin_ms,
        "mean_ci95_upper_ms": mean_ms + ci_margin_ms,
        "per_image_latency_ms": mean_ms / batch_size,
        "throughput_images_per_second": batch_size / (mean_ms / 1000.0),
    }


def print_statistics(times_ms: list[float]) -> None:
    stats = summarize_latency(times_ms, batch_size=1)
    print(f"Mean: {stats['mean_batch_latency_ms']:.4f} ms")
    print(f"Median: {stats['median_batch_latency_ms']:.4f} ms")
    print(f"P95: {stats['p95_batch_latency_ms']:.4f} ms")
    print(f"P99: {stats['p99_batch_latency_ms']:.4f} ms")
    print(f"Min: {stats['min_batch_latency_ms']:.4f} ms")
    print(f"Max: {stats['max_batch_latency_ms']:.4f} ms")
    print(f"Stddev: {stats['sample_stdev_ms']:.4f} ms")
    print(
        "95% CI: "
        f"[{stats['mean_ci95_lower_ms']:.4f}, "
        f"{stats['mean_ci95_upper_ms']:.4f}] ms"
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_raw_latencies(
    path: Path,
    *,
    metadata: dict[str, Any],
    sections: dict[str, list[float]],
) -> None:
    payload: dict[str, Any] = dict(metadata)
    payload["latencies_ms"] = {
        name: [float(value) for value in values]
        for name, values in sections.items()
    }
    save_json(path, payload)


def safe_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned or "single"
