#!/usr/bin/env python3
"""Analyze CPU benchmark results across Batch and thread-count conditions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MISSING = "missing"
BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT_DIR = BENCHMARK_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "results"
CSV_PATH = RESULTS_DIR / "cpu_summary.csv"
REPORT_PATH = ROOT_DIR / "docs" / "cpu_benchmark_report.md"
ENVIRONMENT_PATH = RESULTS_DIR / "environment" / "cpu_environment.json"

RESULT_PATTERNS = (
    RESULTS_DIR / "pytorch" / "pytorch_metrics_b*_t*.json",
    RESULTS_DIR / "onnx" / "onnx_metrics_b*_t*.json",
)
VALIDATION_PATTERN = RESULTS_DIR / "validation" / "validation_b*_t*.json"
EXPORT_PATTERN = RESULTS_DIR / "onnx_export" / "onnx_export_b*.json"

CSV_FIELDS = [
    "runtime",
    "batch_size",
    "thread_count",
    "inter_op_thread_count",
    "warmup_count",
    "measurement_count",
    "seed",
    "input_shape",
    "dtype",
    "model",
    "provider_or_device",
    "source_file",
    "raw_latency_path",
    "raw_latency_count",
    "initialization_type",
    "initialization_ms",
    "input_loading_ms",
    "tensor_conversion_ms",
    "first_inference_ms",
    "mean_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "std_dev_latency_ms",
    "ci95_lower_latency_ms",
    "ci95_upper_latency_ms",
    "ci95_half_width_latency_ms",
    "per_image_latency_ms",
    "throughput_samples_per_s",
    "onnx_speedup",
    "thread_latency_speedup",
    "rss_before_initialization_mb",
    "rss_after_initialization_mb",
    "initialization_rss_increase_mb",
    "final_rss_mb",
    "peak_rss_mb",
    "validation_max_difference",
    "validation_mean_difference",
    "validation_allclose",
    "validation_top1_match",
]


def first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return data


def normalize_runtime(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).lower()
    if "pytorch" in text or text == "torch":
        return "pytorch_cpu"
    if "onnx" in text or text == "ort":
        return "onnxruntime_cpu"
    return None


def parse_batch_thread(path: Path) -> tuple[int | None, int | None]:
    batch_match = re.search(r"(?:^|_)b(?:atch)?(\d+)(?:_|$)", path.stem, re.IGNORECASE)
    thread_match = re.search(r"(?:^|_)t(?:hreads?)?(\d+)(?:_|$)", path.stem, re.IGNORECASE)
    batch = int(batch_match.group(1)) if batch_match else None
    threads = int(thread_match.group(1)) if thread_match else None
    return batch, threads


def relative_source(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_repo_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return ROOT_DIR / candidate


def raw_latency_count(value: Any) -> int | None:
    path = resolve_repo_path(value)
    if path is None or not path.exists():
        return None
    try:
        data = load_json(path)
        latencies = data.get("latencies_ms")
        return len(latencies) if isinstance(latencies, list) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def derived_throughput(batch_size: int, mean_latency_ms: Any) -> float | None:
    if not isinstance(mean_latency_ms, (int, float)) or mean_latency_ms <= 0:
        return None
    return batch_size * 1000.0 / mean_latency_ms


def derived_speedup(baseline: Any, current: Any) -> float | None:
    if not isinstance(baseline, (int, float)):
        return None
    if not isinstance(current, (int, float)) or current <= 0:
        return None
    return baseline / current


def extract_identity(path: Path, data: dict[str, Any]) -> tuple[str, int, int]:
    file_batch, file_threads = parse_batch_thread(path)
    runtime = normalize_runtime(first(data, "runtime", "framework")) or normalize_runtime(path.stem)
    json_batch = first(data, "batch_size", "batch")
    json_threads = first(data, "intra_op_threads", "thread_count", "num_threads")

    batch = int(json_batch) if json_batch is not None else file_batch
    threads = int(json_threads) if json_threads is not None else file_threads

    if runtime is None or batch is None or threads is None:
        raise ValueError(f"{path}: runtime, batch, or thread count is missing")
    if file_batch is not None and file_batch != batch:
        raise ValueError(f"{path}: filename/JSON batch mismatch")
    if file_threads is not None and file_threads != threads:
        raise ValueError(f"{path}: filename/JSON thread mismatch")
    return runtime, batch, threads


def parse_result(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    runtime, batch_size, threads = extract_identity(path, data)
    is_pytorch = runtime == "pytorch_cpu"
    before = first(
        data,
        "rss_before_model_mb" if is_pytorch else "rss_before_session_mb",
        "rss_before_initialization_mb",
    )
    after = first(
        data,
        "rss_after_model_mb" if is_pytorch else "rss_after_session_mb",
        "rss_after_initialization_mb",
    )
    rss_increase = first(
        data,
        "model_rss_delta_mb" if is_pytorch else "session_rss_delta_mb",
        "initialization_rss_increase_mb",
    )
    if rss_increase is None and isinstance(before, (int, float)) and isinstance(after, (int, float)):
        rss_increase = after - before

    mean_latency = first(data, "mean_batch_latency_ms", "mean_latency_ms")
    throughput = first(
        data,
        "throughput_images_per_second",
        "throughput_samples_per_s",
        "throughput",
    )
    if throughput is None:
        throughput = derived_throughput(batch_size, mean_latency)

    raw_path = first(data, "raw_latency_path")
    return {
        "runtime": runtime,
        "batch_size": batch_size,
        "thread_count": threads,
        "inter_op_thread_count": first(data, "inter_op_threads", "inter_op_thread_count"),
        "warmup_count": first(data, "warmup_count", "warm_up_count", "warmups"),
        "measurement_count": first(data, "measurement_count", "repeat_count", "iterations"),
        "seed": first(data, "seed", "random_seed"),
        "input_shape": first(data, "input_shape"),
        "dtype": first(data, "input_dtype", "dtype"),
        "model": first(data, "model", "model_name"),
        "provider_or_device": first(data, "device") if is_pytorch else first(data, "provider"),
        "source_file": relative_source(path),
        "raw_latency_path": raw_path,
        "raw_latency_count": raw_latency_count(raw_path),
        "initialization_type": "model_initialization" if is_pytorch else "session_loading",
        "initialization_ms": first(
            data,
            "model_initialization_ms" if is_pytorch else "session_load_ms",
            "initialization_ms",
        ),
        "input_loading_ms": first(data, "input_load_ms", "input_loading_ms"),
        "tensor_conversion_ms": first(data, "tensor_conversion_ms", "numpy_to_tensor_ms"),
        "first_inference_ms": first(data, "first_inference_ms"),
        "mean_latency_ms": mean_latency,
        "median_latency_ms": first(data, "median_batch_latency_ms", "median_latency_ms"),
        "p95_latency_ms": first(data, "p95_batch_latency_ms", "p95_latency_ms"),
        "p99_latency_ms": first(data, "p99_batch_latency_ms", "p99_latency_ms"),
        "min_latency_ms": first(data, "min_batch_latency_ms", "min_latency_ms"),
        "max_latency_ms": first(data, "max_batch_latency_ms", "max_latency_ms"),
        "std_dev_latency_ms": first(data, "std_dev_batch_latency_ms", "std_dev_latency_ms"),
        "ci95_lower_latency_ms": first(data, "ci95_lower_batch_latency_ms", "ci95_lower_latency_ms"),
        "ci95_upper_latency_ms": first(data, "ci95_upper_batch_latency_ms", "ci95_upper_latency_ms"),
        "ci95_half_width_latency_ms": first(
            data,
            "ci95_half_width_batch_latency_ms",
            "ci95_half_width_latency_ms",
        ),
        "per_image_latency_ms": first(data, "per_image_latency_ms", "per_sample_latency_ms"),
        "throughput_samples_per_s": throughput,
        "onnx_speedup": None,
        "thread_latency_speedup": None,
        "rss_before_initialization_mb": before,
        "rss_after_initialization_mb": after,
        "initialization_rss_increase_mb": rss_increase,
        "final_rss_mb": first(data, "final_rss_mb"),
        "peak_rss_mb": first(data, "peak_rss_mb"),
        "validation_max_difference": None,
        "validation_mean_difference": None,
        "validation_allclose": None,
        "validation_top1_match": None,
    }


def parse_validation(path: Path, data: dict[str, Any]) -> tuple[tuple[int, int], dict[str, Any]]:
    file_batch, file_threads = parse_batch_thread(path)
    batch = int(first(data, "batch_size", "batch") or file_batch or 0)
    threads = int(first(data, "intra_op_threads", "thread_count") or file_threads or 0)
    if batch <= 0 or threads <= 0:
        raise ValueError(f"{path}: validation batch or thread count is missing")
    return (batch, threads), {
        "validation_max_difference": first(data, "max_absolute_difference", "max_difference"),
        "validation_mean_difference": first(data, "mean_absolute_difference", "mean_difference"),
        "validation_allclose": first(data, "outputs_all_close", "allclose"),
        "validation_top1_match": first(data, "same_top1", "top1_match"),
    }


def collect_rows() -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    result_paths = sorted(path for pattern in RESULT_PATTERNS for path in pattern.parent.glob(pattern.name))
    if not result_paths:
        raise FileNotFoundError("No CPU metric JSON files found")

    rows = [parse_result(path, load_json(path)) for path in result_paths]
    validations: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(VALIDATION_PATTERN.parent.glob(VALIDATION_PATTERN.name)):
        key, validation = parse_validation(path, load_json(path))
        if key in validations:
            raise ValueError(f"duplicate validation result for batch/thread {key}")
        validations[key] = validation

    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["runtime"], row["batch_size"], row["thread_count"])
        if key in indexed:
            raise ValueError(f"duplicate result for runtime/batch/thread {key}")
        indexed[key] = row
        row.update(validations.get((row["batch_size"], row["thread_count"]), {}))

    combinations = sorted({(row["batch_size"], row["thread_count"]) for row in rows})
    for batch, threads in combinations:
        pytorch = indexed.get(("pytorch_cpu", batch, threads))
        onnx = indexed.get(("onnxruntime_cpu", batch, threads))
        if pytorch and onnx:
            onnx["onnx_speedup"] = derived_speedup(
                pytorch["mean_latency_ms"], onnx["mean_latency_ms"]
            )

    for runtime in ("pytorch_cpu", "onnxruntime_cpu"):
        for batch in sorted({row["batch_size"] for row in rows}):
            candidates = [row for row in rows if row["runtime"] == runtime and row["batch_size"] == batch]
            if not candidates:
                continue
            baseline = min(candidates, key=lambda row: row["thread_count"])
            for row in candidates:
                row["thread_latency_speedup"] = derived_speedup(
                    baseline["mean_latency_ms"], row["mean_latency_ms"]
                )

    rows.sort(key=lambda row: (row["thread_count"], row["batch_size"], row["runtime"]))
    return rows, validations


def collect_exports() -> list[dict[str, Any]]:
    exports: list[dict[str, Any]] = []
    for path in sorted(EXPORT_PATTERN.parent.glob(EXPORT_PATTERN.name)):
        data = load_json(path)
        exports.append(
            {
                "batch": first(data, "batch_size"),
                "opset": first(data, "opset_version", "opset"),
                "export_time_ms": first(data, "export_time_ms"),
                "model_size_mb": first(data, "model_size_mb"),
                "checker_passed": first(data, "checker_passed"),
                "node_count": first(data, "node_count"),
                "operator_counts": first(data, "operator_counts"),
            }
        )
    return exports


def collect_environment() -> dict[str, Any]:
    if not ENVIRONMENT_PATH.exists():
        return {}
    return load_json(ENVIRONMENT_PATH)


def csv_value(value: Any) -> Any:
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(rows: list[dict[str, Any]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: csv_value(row.get(field)) for field in CSV_FIELDS}
            for row in rows
        )


def number(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else MISSING


def boolean(value: Any) -> str:
    if value is None:
        return MISSING
    return "통과" if value else "실패"


def runtime_label(runtime: str) -> str:
    return "PyTorch CPU" if runtime == "pytorch_cpu" else "ONNX Runtime CPU"


def pct_change(value: Any, baseline: Any) -> str:
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)) or baseline == 0:
        return MISSING
    return f"{(value / baseline - 1) * 100:+.1f}%"


def md_table(headers: list[str], body: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def report_text(
    rows: list[dict[str, Any]],
    validations: dict[tuple[int, int], dict[str, Any]],
    exports: list[dict[str, Any]],
    environment: dict[str, Any],
) -> str:
    batches = sorted({row["batch_size"] for row in rows})
    threads = sorted({row["thread_count"] for row in rows})
    indexed = {
        (row["runtime"], row["batch_size"], row["thread_count"]): row
        for row in rows
    }

    all_results = [
        [
            runtime_label(row["runtime"]),
            str(row["thread_count"]),
            str(row["batch_size"]),
            number(row["mean_latency_ms"]),
            number(row["median_latency_ms"]),
            number(row["p95_latency_ms"]),
            number(row["p99_latency_ms"]),
            number(row["std_dev_latency_ms"]),
            f"[{number(row['ci95_lower_latency_ms'])}, {number(row['ci95_upper_latency_ms'])}]",
            number(row["per_image_latency_ms"]),
            number(row["throughput_samples_per_s"]),
            number(row["onnx_speedup"]),
            str(row["raw_latency_count"]) if row["raw_latency_count"] is not None else MISSING,
        ]
        for row in rows
    ]

    runtime_comparisons: list[list[str]] = []
    for threads_value in threads:
        for batch in batches:
            pytorch = indexed.get(("pytorch_cpu", batch, threads_value), {})
            onnx = indexed.get(("onnxruntime_cpu", batch, threads_value), {})
            if not pytorch and not onnx:
                continue
            runtime_comparisons.append(
                [
                    str(threads_value),
                    str(batch),
                    number(pytorch.get("mean_latency_ms")),
                    number(onnx.get("mean_latency_ms")),
                    number(onnx.get("onnx_speedup")),
                    number(pytorch.get("throughput_samples_per_s")),
                    number(onnx.get("throughput_samples_per_s")),
                ]
            )

    thread_rows: list[list[str]] = []
    for runtime in ("pytorch_cpu", "onnxruntime_cpu"):
        for batch in batches:
            candidates = [row for row in rows if row["runtime"] == runtime and row["batch_size"] == batch]
            if not candidates:
                continue
            baseline = min(candidates, key=lambda row: row["thread_count"])
            for row in sorted(candidates, key=lambda item: item["thread_count"]):
                thread_rows.append(
                    [
                        runtime_label(runtime),
                        str(batch),
                        str(row["thread_count"]),
                        number(row["mean_latency_ms"]),
                        number(row["thread_latency_speedup"]),
                        pct_change(row["throughput_samples_per_s"], baseline["throughput_samples_per_s"]),
                        number(row["p99_latency_ms"]),
                    ]
                )

    batch_rows: list[list[str]] = []
    for runtime in ("pytorch_cpu", "onnxruntime_cpu"):
        for threads_value in threads:
            candidates = [row for row in rows if row["runtime"] == runtime and row["thread_count"] == threads_value]
            if not candidates:
                continue
            baseline = min(candidates, key=lambda row: row["batch_size"])
            for row in sorted(candidates, key=lambda item: item["batch_size"]):
                batch_rows.append(
                    [
                        runtime_label(runtime),
                        str(threads_value),
                        str(row["batch_size"]),
                        number(row["mean_latency_ms"]),
                        pct_change(row["per_image_latency_ms"], baseline["per_image_latency_ms"]),
                        pct_change(row["throughput_samples_per_s"], baseline["throughput_samples_per_s"]),
                    ]
                )

    validation_rows = [
        [
            str(batch),
            str(threads_value),
            number(values.get("validation_max_difference"), 9),
            number(values.get("validation_mean_difference"), 9),
            boolean(values.get("validation_allclose")),
            boolean(values.get("validation_top1_match")),
        ]
        for (batch, threads_value), values in sorted(validations.items(), key=lambda item: (item[0][1], item[0][0]))
    ]

    export_rows = [
        [
            str(item.get("batch", MISSING)),
            str(item.get("opset", MISSING)),
            number(item.get("export_time_ms")),
            number(item.get("model_size_mb")),
            boolean(item.get("checker_passed")),
            str(item.get("node_count", MISSING)),
            json.dumps(item.get("operator_counts"), ensure_ascii=False, separators=(",", ":"))
            if isinstance(item.get("operator_counts"), dict)
            else MISSING,
        ]
        for item in exports
    ]

    package_versions = environment.get("package_versions", {}) if isinstance(environment, dict) else {}
    env_rows = [
        ["Hostname", str(environment.get("hostname", MISSING))],
        ["OS", str(environment.get("os", MISSING))],
        ["Kernel", str(environment.get("kernel", MISSING))],
        ["Architecture", str(environment.get("architecture", MISSING))],
        ["CPU", str(environment.get("cpu_model", MISSING))],
        ["Physical / Logical cores", f"{environment.get('physical_cpu_cores', MISSING)} / {environment.get('logical_cpu_cores', MISSING)}"],
        ["RAM (GB)", str(environment.get("ram_gb", MISSING))],
        ["Python", str(environment.get("python_version", MISSING))],
        ["PyTorch", str(package_versions.get("torch", MISSING))],
        ["Torchvision", str(package_versions.get("torchvision", MISSING))],
        ["ONNX", str(package_versions.get("onnx", MISSING))],
        ["ONNX Runtime", str(package_versions.get("onnxruntime", MISSING))],
        ["Virtual environment", str(environment.get("virtual_environment", MISSING))],
        ["Git commit", str(environment.get("git_commit", MISSING))],
    ]

    expected_validation_keys = {(batch, thread) for batch in batches for thread in threads}
    validation_passed = (
        expected_validation_keys.issubset(validations.keys())
        and all(
            validations[key].get("validation_allclose") is True
            and validations[key].get("validation_top1_match") is True
            for key in expected_validation_keys
        )
    )

    missing_raw = [row for row in rows if row["raw_latency_count"] != row["measurement_count"]]
    missing_stats = [
        row for row in rows
        if any(row.get(key) is None for key in (
            "p99_latency_ms",
            "std_dev_latency_ms",
            "ci95_lower_latency_ms",
            "ci95_upper_latency_ms",
        ))
    ]

    best_by_condition: list[str] = []
    for thread in threads:
        for batch in batches:
            candidates = [
                row for row in rows
                if row["thread_count"] == thread
                and row["batch_size"] == batch
                and isinstance(row["mean_latency_ms"], (int, float))
            ]
            if candidates:
                best = min(candidates, key=lambda row: row["mean_latency_ms"])
                best_by_condition.append(
                    f"- T{thread}, B{batch}: {runtime_label(best['runtime'])} mean {number(best['mean_latency_ms'])} ms"
                )

    lines = [
        "# CPU Benchmark Report",
        "",
        "## 데이터 범위",
        "",
        f"- 분석 Batch: {', '.join(map(str, batches))}",
        f"- 분석 Intra-op thread: {', '.join(map(str, threads))}",
        f"- 성능 결과: {len(rows)}개",
        f"- Validation 결과: {len(validations)}개",
        f"- ONNX Export 결과: {len(exports)}개",
        "",
        "## 재현 환경",
        "",
        md_table(["항목", "기록값"], env_rows),
        "",
        "## 전체 측정 결과",
        "",
        "Latency 단위는 ms, Throughput 단위는 images/s다. ONNX speedup은 동일 Batch·Thread 조건에서 PyTorch mean / ONNX mean이다.",
        "",
        md_table(
            [
                "Runtime",
                "Threads",
                "Batch",
                "Mean",
                "Median",
                "P95",
                "P99",
                "Std",
                "95% CI",
                "Per-image",
                "Throughput",
                "ONNX speedup",
                "Raw count",
            ],
            all_results,
        ),
        "",
        "## Runtime 비교",
        "",
        md_table(
            [
                "Threads",
                "Batch",
                "PyTorch mean",
                "ONNX mean",
                "ONNX speedup",
                "PyTorch images/s",
                "ONNX images/s",
            ],
            runtime_comparisons,
        ),
        "",
        "## Thread 확장성",
        "",
        "Thread latency speedup은 각 Runtime·Batch에서 가장 작은 Thread 수를 기준으로 계산한다.",
        "",
        md_table(
            [
                "Runtime",
                "Batch",
                "Threads",
                "Mean",
                "Latency speedup",
                "Throughput 변화",
                "P99",
            ],
            thread_rows,
        ),
        "",
        "## Batch 확장성",
        "",
        "Per-image와 Throughput 변화는 동일 Runtime·Thread에서 가장 작은 Batch를 기준으로 계산한다.",
        "",
        md_table(
            ["Runtime", "Threads", "Batch", "Mean", "Per-image 변화", "Throughput 변화"],
            batch_rows,
        ),
        "",
        "## ONNX Export 기록",
        "",
        md_table(
            ["Batch", "Opset", "Export ms", "Size MB", "Checker", "Nodes", "Operator counts"],
            export_rows,
        ) if export_rows else "ONNX Export 메타데이터가 없다.",
        "",
        "## 출력 검증",
        "",
        md_table(
            ["Batch", "Threads", "Max diff", "Mean diff", "Allclose", "Top-1"],
            validation_rows,
        ) if validation_rows else "Validation 결과가 없다.",
        "",
        f"전체 Batch·Thread 검증 결과: **{'전체 통과' if validation_passed else '누락 또는 실패 있음'}**",
        "",
        "## 조건별 최저 Mean Runtime",
        "",
        *best_by_condition,
        "",
        "## 데이터 품질 점검",
        "",
        f"- 원시 Latency 개수 불일치 또는 누락: {len(missing_raw)}개",
        f"- P99·표준편차·95% CI 누락: {len(missing_stats)}개",
        f"- 기대 Validation 조합: {len(expected_validation_keys)}개, 실제: {len(validations)}개",
        "",
        "## 해석 시 주의사항",
        "",
        "- Thread 수가 많다고 항상 Mean, P99, Throughput이 개선되는 것은 아니다.",
        "- Runtime 우위는 Batch와 Thread 조건을 고정한 뒤 비교해야 한다.",
        "- 95% CI는 한 실행 안의 iteration 표본에 대한 구간이며, 독립 프로세스 반복 간 재현성을 대신하지 않는다.",
        "- RSS는 전체 프로세스와 Model/Session 생성 증가분을 구분해서 해석해야 한다.",
        "- ONNX Checker 통과는 실행 가능성 검증이지 성능 향상 보장이 아니다.",
        "",
        "## 다음 단계",
        "",
        "1. 같은 Batch·Thread 조합을 독립 프로세스로 여러 번 반복해 실행 간 분산을 측정한다.",
        "2. CPU affinity, 전원 정책, 백그라운드 부하를 기록한다.",
        "3. Thread별 CPU utilization을 함께 수집한다.",
        "4. Fixed Shape와 Dynamic Shape 비교로 확장한다.",
        "",
    ]
    return "\n".join(lines)


def write_report(
    rows: list[dict[str, Any]],
    validations: dict[tuple[int, int], dict[str, Any]],
    exports: list[dict[str, Any]],
    environment: dict[str, Any],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        report_text(rows, validations, exports, environment),
        encoding="utf-8",
    )


def self_check() -> None:
    batch, threads = parse_batch_thread(Path("onnx_metrics_b16_t4.json"))
    assert batch == 16 and threads == 4

    fixture = {
        "runtime": "pytorch",
        "device": "cpu",
        "batch_size": 4,
        "intra_op_threads": 2,
        "inter_op_threads": 1,
        "warmup_count": 10,
        "measurement_count": 300,
        "mean_batch_latency_ms": 50.0,
        "median_batch_latency_ms": 49.0,
        "p95_batch_latency_ms": 55.0,
        "p99_batch_latency_ms": 58.0,
        "std_dev_batch_latency_ms": 2.0,
        "ci95_lower_batch_latency_ms": 49.77,
        "ci95_upper_batch_latency_ms": 50.23,
        "per_image_latency_ms": 12.5,
        "throughput_images_per_second": 80.0,
    }
    parsed = parse_result(Path("pytorch_metrics_b4_t2.json"), fixture)
    assert parsed["runtime"] == "pytorch_cpu"
    assert parsed["batch_size"] == 4
    assert parsed["thread_count"] == 2
    assert math.isclose(parsed["throughput_samples_per_s"], 80.0)
    assert math.isclose(derived_speedup(20.0, 10.0) or 0.0, 2.0)

    key, validation = parse_validation(
        Path("validation_b4_t2.json"),
        {
            "batch_size": 4,
            "intra_op_threads": 2,
            "outputs_all_close": True,
            "same_top1": True,
        },
    )
    assert key == (4, 2)
    assert validation["validation_allclose"] is True
    print("self-check: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    rows, validations = collect_rows()
    exports = collect_exports()
    environment = collect_environment()
    write_csv(rows)
    write_report(rows, validations, exports, environment)
    print(f"wrote {relative_source(CSV_PATH)}")
    print(f"wrote {relative_source(REPORT_PATH)}")


if __name__ == "__main__":
    main()
