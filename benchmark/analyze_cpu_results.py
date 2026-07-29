#!/usr/bin/env python3
"""Summarize CPU benchmark JSON files without modifying the source data."""

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

RESULT_GLOBS = (
    RESULTS_DIR / "pytorch" / "*.json",
    RESULTS_DIR / "onnx" / "*.json",
)
VALIDATION_GLOB = RESULTS_DIR / "validation" / "*.json"

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
    "initialization_type",
    "initialization_ms",
    "input_loading_ms",
    "tensor_conversion_ms",
    "first_inference_ms",
    "mean_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "per_image_latency_ms",
    "throughput_samples_per_s",
    "onnx_speedup",
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


def normalize_runtime(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).lower()
    if "pytorch" in text or text == "torch":
        return "pytorch_cpu"
    if "onnx" in text or text == "ort":
        return "onnxruntime_cpu"
    return None


def identity_from_filename(path: Path) -> tuple[str | None, int | None]:
    runtime = normalize_runtime(path.stem)
    match = re.search(r"(?:^|_)b(?:atch)?(\d+)(?:_|$)", path.stem, re.IGNORECASE)
    return runtime, int(match.group(1)) if match else None


def extract_identity(path: Path, data: dict[str, Any]) -> tuple[str, int]:
    file_runtime, file_batch = identity_from_filename(path)
    json_runtime = normalize_runtime(first(data, "runtime", "framework"))
    json_batch = first(data, "batch_size", "batch")

    if file_runtime and json_runtime and file_runtime != json_runtime:
        raise ValueError(f"{path}: filename/JSON runtime mismatch")
    if file_batch is not None and json_batch is not None and file_batch != int(json_batch):
        raise ValueError(f"{path}: filename/JSON batch mismatch")

    runtime = json_runtime or file_runtime
    batch = int(json_batch) if json_batch is not None else file_batch
    if runtime is None or batch is None:
        raise ValueError(f"{path}: runtime or batch size is missing")
    return runtime, batch


def derived_throughput(batch_size: int, mean_latency_ms: Any) -> float | None:
    if not isinstance(mean_latency_ms, (int, float)) or mean_latency_ms <= 0:
        return None
    return batch_size / (mean_latency_ms / 1000)


def derived_speedup(pytorch_mean_ms: Any, onnx_mean_ms: Any) -> float | None:
    if not isinstance(pytorch_mean_ms, (int, float)):
        return None
    if not isinstance(onnx_mean_ms, (int, float)) or onnx_mean_ms <= 0:
        return None
    return pytorch_mean_ms / onnx_mean_ms


def relative_source(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def parse_result(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    runtime, batch_size = extract_identity(path, data)
    is_pytorch = runtime == "pytorch_cpu"
    initialization_type = "model_initialization" if is_pytorch else "session_loading"

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

    row = {
        "runtime": runtime,
        "batch_size": batch_size,
        "thread_count": first(data, "intra_op_threads", "thread_count", "num_threads"),
        "inter_op_thread_count": first(data, "inter_op_threads", "inter_op_thread_count"),
        "warmup_count": first(data, "warmup_count", "warm_up_count", "warmups"),
        "measurement_count": first(data, "measurement_count", "repeat_count", "iterations"),
        "seed": first(data, "seed", "random_seed"),
        "input_shape": first(data, "input_shape"),
        "dtype": first(data, "input_dtype", "dtype"),
        "model": first(data, "model", "model_name"),
        "provider_or_device": first(data, "device") if is_pytorch else first(data, "provider"),
        "source_file": relative_source(path),
        "initialization_type": initialization_type,
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
        "min_latency_ms": first(data, "min_batch_latency_ms", "min_latency_ms"),
        "max_latency_ms": first(data, "max_batch_latency_ms", "max_latency_ms"),
        "per_image_latency_ms": first(
            data, "per_image_latency_ms", "per_sample_latency_ms"
        ),
        "throughput_samples_per_s": throughput,
        "onnx_speedup": None,
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
    return row


def parse_validation(path: Path, data: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    _, file_batch = identity_from_filename(path)
    json_batch = first(data, "batch_size", "batch")
    if file_batch is not None and json_batch is not None and file_batch != int(json_batch):
        raise ValueError(f"{path}: filename/JSON batch mismatch")
    batch = int(json_batch) if json_batch is not None else file_batch
    if batch is None:
        raise ValueError(f"{path}: validation batch size is missing")
    return batch, {
        "validation_max_difference": first(
            data, "max_absolute_difference", "max_difference"
        ),
        "validation_mean_difference": first(
            data, "mean_absolute_difference", "mean_difference"
        ),
        "validation_allclose": first(data, "outputs_all_close", "allclose"),
        "validation_top1_match": first(data, "same_top1", "top1_match"),
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def collect_rows() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    result_paths = sorted(path for pattern in RESULT_GLOBS for path in pattern.parent.glob(pattern.name))
    validation_paths = sorted(VALIDATION_GLOB.parent.glob(VALIDATION_GLOB.name))
    if not result_paths:
        raise FileNotFoundError("No PyTorch or ONNX result JSON files found")

    rows = [parse_result(path, load_json(path)) for path in result_paths]
    validations: dict[int, dict[str, Any]] = {}
    for path in validation_paths:
        batch, validation = parse_validation(path, load_json(path))
        if batch in validations:
            raise ValueError(f"duplicate validation result for batch {batch}")
        validations[batch] = validation

    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["runtime"], row["batch_size"])
        if key in indexed:
            raise ValueError(f"duplicate result for runtime/batch {key}")
        indexed[key] = row
        row.update(validations.get(row["batch_size"], {}))

    batches = sorted({row["batch_size"] for row in rows})
    for batch in batches:
        pytorch = indexed.get(("pytorch_cpu", batch))
        onnx = indexed.get(("onnxruntime_cpu", batch))
        if pytorch and onnx:
            onnx["onnx_speedup"] = derived_speedup(
                pytorch["mean_latency_ms"], onnx["mean_latency_ms"]
            )

    rows.sort(key=lambda row: (row["batch_size"], row["runtime"]))
    return rows, validations


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
            {field: csv_value(row.get(field)) for field in CSV_FIELDS} for row in rows
        )


def number(value: Any, digits: int = 3) -> str:
    if not isinstance(value, (int, float)):
        return MISSING
    return f"{value:.{digits}f}"


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


def ratio(value: Any, baseline: Any) -> str:
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)) or baseline == 0:
        return MISSING
    return f"{value / baseline:.3f}x"


def md_table(headers: list[str], body: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def report_text(
    rows: list[dict[str, Any]], validations: dict[int, dict[str, Any]]
) -> str:
    indexed = {(row["runtime"], row["batch_size"]): row for row in rows}
    batches = sorted({row["batch_size"] for row in rows})
    speedups = {
        batch: indexed.get(("onnxruntime_cpu", batch), {}).get("onnx_speedup")
        for batch in batches
    }

    all_results = []
    for row in rows:
        all_results.append(
            [
                runtime_label(row["runtime"]),
                str(row["batch_size"]),
                number(row["mean_latency_ms"]),
                number(row["median_latency_ms"]),
                number(row["p95_latency_ms"]),
                number(row["min_latency_ms"]),
                number(row["max_latency_ms"]),
                number(row["per_image_latency_ms"]),
                number(row["throughput_samples_per_s"]),
                number(row["onnx_speedup"]),
                number(row["initialization_ms"]),
                number(row["first_inference_ms"]),
            ]
        )

    comparisons = []
    scalability = []
    for batch in batches:
        pytorch = indexed.get(("pytorch_cpu", batch), {})
        onnx = indexed.get(("onnxruntime_cpu", batch), {})
        comparisons.append(
            [
                str(batch),
                number(pytorch.get("mean_latency_ms")),
                number(onnx.get("mean_latency_ms")),
                number(speedups.get(batch)),
                number(pytorch.get("throughput_samples_per_s")),
                number(onnx.get("throughput_samples_per_s")),
            ]
        )
        for runtime in ("pytorch_cpu", "onnxruntime_cpu"):
            current = indexed.get((runtime, batch), {})
            baseline = indexed.get((runtime, batches[0]), {})
            scalability.append(
                [
                    runtime_label(runtime),
                    str(batch),
                    ratio(current.get("mean_latency_ms"), baseline.get("mean_latency_ms")),
                    pct_change(
                        current.get("per_image_latency_ms"),
                        baseline.get("per_image_latency_ms"),
                    ),
                    pct_change(
                        current.get("throughput_samples_per_s"),
                        baseline.get("throughput_samples_per_s"),
                    ),
                    number(speedups.get(batch))
                    if runtime == "onnxruntime_cpu"
                    else MISSING,
                ]
            )

    initialization = [
        [
            runtime_label(row["runtime"]),
            str(row["batch_size"]),
            row["initialization_type"],
            number(row["initialization_ms"]),
            number(row["input_loading_ms"]),
            number(row["tensor_conversion_ms"]),
            number(row["first_inference_ms"]),
            ratio(row["first_inference_ms"], row["mean_latency_ms"]),
        ]
        for row in rows
    ]
    memory = [
        [
            runtime_label(row["runtime"]),
            str(row["batch_size"]),
            number(row["rss_before_initialization_mb"]),
            number(row["rss_after_initialization_mb"]),
            number(row["initialization_rss_increase_mb"]),
            number(row["final_rss_mb"]),
            number(row["peak_rss_mb"]),
        ]
        for row in rows
    ]
    validation_rows = [
        [
            str(batch),
            number(values.get("validation_max_difference"), 9),
            number(values.get("validation_mean_difference"), 9),
            boolean(values.get("validation_allclose")),
            boolean(values.get("validation_top1_match")),
        ]
        for batch, values in sorted(validations.items())
    ]

    common = rows[0]
    shapes = ", ".join(
        f"B{batch}: {json.dumps(indexed[next(key for key in indexed if key[1] == batch)]['input_shape'], separators=(',', ':'))}"
        for batch in batches
    )
    dtype_values = sorted(
        {str(row["dtype"]) for row in rows if row.get("dtype") is not None}
    )
    model_values = sorted(
        {str(row["model"]) for row in rows if row.get("model") is not None}
    )
    seeds = sorted({str(row["seed"]) for row in rows if row.get("seed") is not None})
    validation_passed = all(
        values.get("validation_allclose") is True
        and values.get("validation_top1_match") is True
        for values in validations.values()
    ) and set(validations) == set(batches)

    pytorch_b1 = indexed.get(("pytorch_cpu", batches[0]), {})
    pytorch_b4 = indexed.get(("pytorch_cpu", 4), {})
    pytorch_b16 = indexed.get(("pytorch_cpu", 16), {})
    onnx_b1 = indexed.get(("onnxruntime_cpu", batches[0]), {})
    onnx_b4 = indexed.get(("onnxruntime_cpu", 4), {})
    onnx_b16 = indexed.get(("onnxruntime_cpu", 16), {})

    lines = [
        "# CPU Benchmark Report",
        "",
        "## 데이터 범위",
        "",
        f"- PyTorch 결과 {sum(row['runtime'] == 'pytorch_cpu' for row in rows)}개: `benchmark/results/pytorch/*.json`",
        f"- ONNX Runtime 결과 {sum(row['runtime'] == 'onnxruntime_cpu' for row in rows)}개: `benchmark/results/onnx/*.json`",
        f"- Validation 결과 {len(validations)}개: `benchmark/results/validation/*.json`",
        f"- 분석 Batch: {', '.join(map(str, batches))}. 원본 JSON은 읽기만 했다.",
        "",
        "## 실험 환경",
        "",
        md_table(
            ["항목", "기록값"],
            [
                ["Model", ", ".join(model_values) if model_values else MISSING],
                ["Runtime", "PyTorch CPU, ONNX Runtime CPU (CPUExecutionProvider)"],
                ["Intra-op threads", str(common["thread_count"]) if common["thread_count"] is not None else MISSING],
                ["Inter-op threads", str(common["inter_op_thread_count"]) if common["inter_op_thread_count"] is not None else MISSING],
                ["Warm-up", str(common["warmup_count"]) if common["warmup_count"] is not None else MISSING],
                ["측정 반복", str(common["measurement_count"]) if common["measurement_count"] is not None else MISSING],
                ["Seed", ", ".join(seeds) if seeds else MISSING],
                ["Input shape", shapes],
                ["Dtype", ", ".join(dtype_values) if dtype_values else MISSING],
                ["CPU 모델 / OS / Runtime 버전", MISSING],
            ],
        ),
        "",
        "환경값은 JSON에 기록된 범위만 사용했다. CPU 모델, OS, PyTorch/ONNX Runtime 버전은 기록되지 않아 추정하지 않았다.",
        "",
        "## 전체 측정 결과 표",
        "",
        "Latency 단위 ms, Throughput 단위 samples/s다. Speedup은 같은 Batch의 `PyTorch mean / ONNX mean`이며 ONNX 행에 표시한다.",
        "",
        md_table(
            [
                "Runtime",
                "Batch",
                "Mean",
                "Median",
                "P95",
                "Min",
                "Max",
                "Per-image",
                "Throughput",
                "ONNX speedup",
                "Init",
                "First",
            ],
            all_results,
        ),
        "",
        "## PyTorch CPU와 ONNX Runtime CPU 비교",
        "",
        md_table(
            [
                "Batch",
                "PyTorch mean (ms)",
                "ONNX mean (ms)",
                "ONNX speedup",
                "PyTorch samples/s",
                "ONNX samples/s",
            ],
            comparisons,
        ),
        "",
        f"ONNX Runtime CPU가 측정된 모든 Batch에서 더 낮은 mean latency를 보였다. Speedup은 B1 {number(speedups.get(1))}x, B4 {number(speedups.get(4))}x, B16 {number(speedups.get(16))}x다. Batch 증가와 함께 이 데이터의 ONNX 이점은 감소했다. 결과 파일만으로 커널 구현, 그래프 최적화, 메모리 접근 등 원인을 확정할 수 없다.",
        "",
        "## Batch 1, 4, 16 확장성",
        "",
        "Mean 배수, Per-image 변화, Throughput 변화는 각 Runtime의 B1 대비다.",
        "",
        md_table(
            [
                "Runtime",
                "Batch",
                "Mean 배수",
                "Per-image 변화",
                "Throughput 변화",
                "ONNX speedup",
            ],
            scalability,
        ),
        "",
        f"PyTorch는 B4에서 per-image latency가 B1 대비 {pct_change(pytorch_b4.get('per_image_latency_ms'), pytorch_b1.get('per_image_latency_ms'))}, throughput은 {pct_change(pytorch_b4.get('throughput_samples_per_s'), pytorch_b1.get('throughput_samples_per_s'))}였다. B16에서는 각각 {pct_change(pytorch_b16.get('per_image_latency_ms'), pytorch_b1.get('per_image_latency_ms'))}, {pct_change(pytorch_b16.get('throughput_samples_per_s'), pytorch_b1.get('throughput_samples_per_s'))}였다.",
        "",
        f"ONNX Runtime은 B4에서 per-image latency가 B1 대비 {pct_change(onnx_b4.get('per_image_latency_ms'), onnx_b1.get('per_image_latency_ms'))}, throughput은 {pct_change(onnx_b4.get('throughput_samples_per_s'), onnx_b1.get('throughput_samples_per_s'))}였다. B16에서는 각각 {pct_change(onnx_b16.get('per_image_latency_ms'), onnx_b1.get('per_image_latency_ms'))}, {pct_change(onnx_b16.get('throughput_samples_per_s'), onnx_b1.get('throughput_samples_per_s'))}였다.",
        "",
        "## Latency와 Throughput 분석",
        "",
        f"PyTorch 최고 throughput은 B4의 {number(pytorch_b4.get('throughput_samples_per_s'))} samples/s다. ONNX Runtime 최고 throughput은 B1의 {number(onnx_b1.get('throughput_samples_per_s'))} samples/s이며 B4는 비슷하지만 B16에서 {number(onnx_b16.get('throughput_samples_per_s'))} samples/s로 낮아졌다. 큰 Batch가 자동으로 더 좋은 per-image latency나 throughput을 만들지 않았다.",
        "",
        "Mean, median, P95, min, max는 JSON의 집계값을 그대로 사용했다. 원시 iteration latency가 없어 분산 형태를 복원할 수 없다.",
        "",
        "## 초기화와 First Inference 분석",
        "",
        md_table(
            [
                "Runtime",
                "Batch",
                "초기화 종류",
                "초기화 (ms)",
                "Input load (ms)",
                "NumPy→Tensor (ms)",
                "First (ms)",
                "First/Mean",
            ],
            initialization,
        ),
        "",
        "PyTorch는 model initialization, ONNX Runtime은 session loading을 별도 항목으로 유지했다. ONNX JSON에 NumPy→Tensor 변환값은 없어 `missing`이다. First inference는 대체로 steady-state mean보다 높지만 ONNX B16은 낮다. 따라서 First inference를 항상 초기 실행 페널티로 해석할 수 없다.",
        "",
        "## Memory와 RSS 분석",
        "",
        md_table(
            [
                "Runtime",
                "Batch",
                "생성 전 RSS",
                "생성 후 RSS",
                "생성 증가분",
                "Final RSS",
                "Peak RSS",
            ],
            memory,
        ),
        "",
        "생성 전/후 차이와 JSON의 model/session RSS delta는 Model 또는 Session 생성 증가분이다. Final/Peak RSS는 전체 프로세스 RSS다. PyTorch와 ONNX의 생성 전 RSS가 크게 다르므로 전체 RSS 차이를 Model 또는 Session 자체 메모리 차이로 간주하면 안 된다. 별도 프로세스의 런타임 기본 메모리와 측정 시점이 포함될 수 있다.",
        "",
        "## Validation 분석",
        "",
        md_table(
            ["Batch", "Max difference", "Mean difference", "Allclose", "Top-1 일치"],
            validation_rows,
        ),
        "",
        f"Batch별 Validation 전체 통과 여부: **{'전체 통과' if validation_passed else '전체 통과 아님'}**. Allclose와 Top-1 일치를 각각 확인했다.",
        "",
        "## 이상치와 가능한 원인",
        "",
        f"- B4 PyTorch max latency {number(pytorch_b4.get('max_latency_ms'))} ms는 median {number(pytorch_b4.get('median_latency_ms'))} ms보다 {pct_change(pytorch_b4.get('max_latency_ms'), pytorch_b4.get('median_latency_ms'))} 높다.",
        f"- B4 ONNX max latency {number(onnx_b4.get('max_latency_ms'))} ms는 median {number(onnx_b4.get('median_latency_ms'))} ms보다 {pct_change(onnx_b4.get('max_latency_ms'), onnx_b4.get('median_latency_ms'))} 높다.",
        f"- ONNX B16 mean {number(onnx_b16.get('mean_latency_ms'))} ms가 median {number(onnx_b16.get('median_latency_ms'))} ms보다 낮고, first inference {number(onnx_b16.get('first_inference_ms'))} ms도 mean보다 낮다.",
        "- 모든 ONNX 결과에서 Final RSS가 기록된 Peak RSS보다 소폭 높다. Peak 측정 구간 또는 측정 시점 정의를 확인할 필요가 있다.",
        "- 가능한 요인은 OS 스케줄링, CPU 주파수 변화, 캐시 상태, 백그라운드 부하, 메모리 할당 등이다. 원시 latency와 환경 로그가 없어 어느 원인도 확정할 수 없다.",
        "",
        "## 실험 한계",
        "",
        "- 원시 iteration latency가 없다. 표준편차, P99, Confidence Interval을 계산하지 않았다.",
        "- CPU 모델, OS, Runtime/라이브러리 버전, 전원 정책, CPU affinity, 백그라운드 부하가 기록되지 않았다.",
        "- Seed가 기록되지 않았다.",
        "- 각 Runtime/Batch 조합이 집계 JSON 1개뿐이다. 실행 간 변동성과 재현성을 평가할 수 없다.",
        "- PyTorch와 ONNX의 초기 프로세스 RSS가 달라 전체 RSS 절대값의 직접 비교가 제한된다.",
        "- 결과는 ResNet-18, float32, thread 4, 입력 크기 224×224 범위에 한정된다.",
        "",
        "## 다음 실험 제안",
        "",
        "1. 원시 iteration latency와 실행별 타임스탬프를 저장하고 독립 실행을 여러 번 반복한다.",
        "2. CPU 모델, OS, PyTorch/ONNX Runtime 버전, 전원 정책, affinity, 동시 부하를 기록한다.",
        "3. Thread 수를 1/2/4/8로 바꾸고 Batch 1/4/16의 latency, throughput, speedup을 다시 측정한다.",
        "4. RSS 샘플링 구간을 명시하고 초기 프로세스, 생성 직후, warm-up 후, 측정 중 peak를 같은 정의로 기록한다.",
        "5. 입력을 여러 seed로 생성하고 각 Batch에서 Allclose와 Top-1 validation을 반복한다.",
        "",
    ]
    return "\n".join(lines)


def write_report(rows: list[dict[str, Any]], validations: dict[int, dict[str, Any]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text(rows, validations), encoding="utf-8")


def self_check() -> None:
    runtime, batch = identity_from_filename(Path("onnx_metrics_b16_t4.json"))
    assert runtime == "onnxruntime_cpu" and batch == 16

    fixture = {
        "runtime": "pytorch",
        "device": "cpu",
        "batch_size": 4,
        "intra_op_threads": 4,
        "warmup_count": 10,
        "measurement_count": 300,
        "input_shape": [4, 3, 224, 224],
        "input_dtype": "float32",
        "model_initialization_ms": 12.5,
        "input_load_ms": 1.25,
        "tensor_conversion_ms": 0.02,
        "first_inference_ms": 20.0,
        "mean_batch_latency_ms": 50.0,
        "median_batch_latency_ms": 49.0,
        "p95_batch_latency_ms": 55.0,
        "min_batch_latency_ms": 45.0,
        "max_batch_latency_ms": 60.0,
        "rss_before_model_mb": 100.0,
        "rss_after_model_mb": 125.0,
        "final_rss_mb": 140.0,
        "peak_rss_mb": 150.0,
    }
    parsed = parse_result(Path("pytorch_metrics_b4_t4.json"), fixture)
    assert parsed["runtime"] == "pytorch_cpu" and parsed["batch_size"] == 4
    assert parsed["initialization_ms"] == 12.5
    assert parsed["input_loading_ms"] == 1.25
    assert parsed["tensor_conversion_ms"] == 0.02
    assert parsed["median_latency_ms"] == 49.0
    assert parsed["initialization_rss_increase_mb"] == 25.0
    assert math.isclose(parsed["throughput_samples_per_s"], 80.0)
    assert math.isclose(derived_throughput(4, 50.0) or 0, 80.0)
    assert math.isclose(derived_speedup(20.0, 10.0) or 0, 2.0)

    validation_batch, validation = parse_validation(
        Path("validation_b4.json"),
        {
            "batch_size": 4,
            "max_absolute_difference": 1e-6,
            "mean_absolute_difference": 1e-7,
            "outputs_all_close": True,
            "same_top1": True,
        },
    )
    assert validation_batch == 4
    assert validation["validation_allclose"] is True
    assert validation["validation_top1_match"] is True
    print("self-check: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run parser and calculation checks without writing output files",
    )
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return

    rows, validations = collect_rows()
    write_csv(rows)
    write_report(rows, validations)
    print(f"wrote {relative_source(CSV_PATH)}")
    print(f"wrote {relative_source(REPORT_PATH)}")


if __name__ == "__main__":
    main()
