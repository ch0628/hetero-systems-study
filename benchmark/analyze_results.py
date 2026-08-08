from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict 
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"
PYTORCH_DIR = BENCHMARK_DIR / "results" / "pytorch"
ONNX_DIR = BENCHMARK_DIR / "results" / "onnx"
VALIDATION_DIR = BENCHMARK_DIR / "results" / "validation"
TELEMETRY_DIR = BENCHMARK_DIR / "results" / "telemetry"
PROFILE_DIR = BENCHMARK_DIR / "results" / "profiles"
RESULT_DIR = BENCHMARK_DIR / "results"
DOCS_DIR = ROOT / "docs"

REPEATED_CSV = RESULT_DIR / "gpu_repeated_runs.csv"
SUMMARY_CSV = RESULT_DIR / "gpu_summary.csv"
TELEMETRY_CSV = RESULT_DIR / "gpu_telemetry_summary.csv"
REPORT_PATH = DOCS_DIR / "gpu_benchmark_report.md"

RUNTIME_LABEL = {
    "pytorch": "PyTorch CUDA",
    "onnxruntime": "ONNX Runtime CUDA",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def filename_batch(path: Path) -> int | None:
    match = re.search(r"(?:^|_)b(\d+)(?:_|$)", path.stem.lower())
    return int(match.group(1)) if match else None


def filename_run_id(path: Path) -> str:
    match = re.search(r"_b\d+_(.+)$", path.stem)
    return match.group(1) if match else "legacy"


def normalize_section(section: dict[str, Any] | None) -> dict[str, float | None]:
    section = section or {}
    return {
        "mean": safe_float(section.get("mean_batch_latency_ms")),
        "median": safe_float(section.get("median_batch_latency_ms")),
        "p95": safe_float(section.get("p95_batch_latency_ms")),
        "p99": safe_float(section.get("p99_batch_latency_ms")),
        "min": safe_float(section.get("min_batch_latency_ms")),
        "max": safe_float(section.get("max_batch_latency_ms")),
        "stdev": safe_float(section.get("sample_stdev_ms")),
        "ci95_lower": safe_float(section.get("mean_ci95_lower_ms")),
        "ci95_upper": safe_float(section.get("mean_ci95_upper_ms")),
        "throughput": safe_float(section.get("throughput_images_per_second")),
        "per_image": safe_float(section.get("per_image_latency_ms")),
    }


def load_metric_records() -> list[dict[str, Any]]:
    paths = sorted(PYTORCH_DIR.glob("pytorch_metrics_cuda_b*.json")) + sorted(
        ONNX_DIR.glob("onnx_metrics_cuda_b*.json")
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        data = read_json(path)
        runtime_key = str(data.get("runtime") or "").lower()
        if runtime_key not in RUNTIME_LABEL:
            runtime_key = "pytorch" if path.name.startswith("pytorch_") else "onnxruntime"

        provider = data.get("device") or data.get("provider_request")
        if str(provider).lower() not in {"cuda", "gpu"}:
            continue

        sections = data.get("sections") or {}
        gpu_only_source = sections.get("gpu_only") or data.get("gpu_only")
        e2e_source = sections.get("end_to_end") or data.get("end_to_end")
        if not gpu_only_source or not e2e_source:
            continue

        batch = data.get("batch_size") or filename_batch(path)
        run_id = str(data.get("run_id") or filename_run_id(path))
        if run_id.startswith("profile_") or run_id.startswith("nsys_"):
            continue
        record = {
            "runtime_key": runtime_key,
            "runtime": RUNTIME_LABEL[runtime_key],
            "batch_size": int(batch),
            "run_id": run_id,
            "run_order": data.get("run_order"),
            "source_file": path.relative_to(ROOT).as_posix(),
            "started_at_utc": data.get("started_at_utc"),
            "gpu_only": normalize_section(gpu_only_source),
            "end_to_end": normalize_section(e2e_source),
            "first_inference_ms": safe_float(data.get("first_inference_ms")),
            "initialization_ms": safe_float(
                data.get("model_initialization_ms")
                if runtime_key == "pytorch"
                else data.get("session_load_ms")
            ),
            "environment": {
                "hostname": data.get("hostname"),
                "model": data.get("model"),
                "precision": data.get("precision"),
                "warmup_count": data.get("warmup_count"),
                "measurement_count": data.get("measurement_count"),
                "seed": data.get("seed"),
                "gpu_name": data.get("gpu_name"),
                "driver": data.get("nvidia_driver_version"),
                "torch": data.get("torch_version"),
                "onnxruntime": data.get("onnxruntime_version"),
                "cuda_runtime": data.get("cuda_runtime"),
                "cudnn": data.get("cudnn_version"),
                "tf32": data.get("tf32") or data.get("tf32_control"),
                "slurm_job_id": data.get("slurm_job_id"),
                "git_commit": data.get("git_commit"),
                "providers": data.get("session_providers"),
            },
        }
        records.append(record)

    repeated_pattern = re.compile(r"^r\d+_o\d+$")
    if any(repeated_pattern.match(record["run_id"]) for record in records):
        records = [
            record
            for record in records
            if repeated_pattern.match(record["run_id"])
        ]
    return records


def mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def stdev_or_zero(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return statistics.stdev(clean) if len(clean) > 1 else 0.0


def ci95(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    clean = [value for value in values if value is not None]
    if not clean:
        return None, None
    mean = statistics.fmean(clean)
    if len(clean) == 1:
        return mean, mean
    margin = 1.96 * statistics.stdev(clean) / math.sqrt(len(clean))
    return mean - margin, mean + margin


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["runtime_key"], record["batch_size"])].append(record)

    rows: list[dict[str, Any]] = []
    for (runtime_key, batch), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        row: dict[str, Any] = {
            "runtime_key": runtime_key,
            "runtime": RUNTIME_LABEL[runtime_key],
            "batch_size": batch,
            "independent_run_count": len(group),
        }
        for section_name in ("gpu_only", "end_to_end"):
            mean_values = [record[section_name]["mean"] for record in group]
            lower, upper = ci95(mean_values)
            prefix = section_name
            row[f"{prefix}_mean_ms"] = mean_or_none(mean_values)
            row[f"{prefix}_run_stdev_ms"] = stdev_or_zero(mean_values)
            row[f"{prefix}_run_mean_ci95_lower_ms"] = lower
            row[f"{prefix}_run_mean_ci95_upper_ms"] = upper
            row[f"{prefix}_p95_mean_ms"] = mean_or_none(
                record[section_name]["p95"] for record in group
            )
            row[f"{prefix}_p99_mean_ms"] = mean_or_none(
                record[section_name]["p99"] for record in group
            )
            row[f"{prefix}_throughput_mean"] = mean_or_none(
                record[section_name]["throughput"] for record in group
            )
        row["first_inference_mean_ms"] = mean_or_none(
            record["first_inference_ms"] for record in group
        )
        row["first_inference_run_stdev_ms"] = stdev_or_zero(
            record["first_inference_ms"] for record in group
        )
        row["initialization_mean_ms"] = mean_or_none(
            record["initialization_ms"] for record in group
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_repeated(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "runtime": record["runtime"],
            "batch_size": record["batch_size"],
            "run_id": record["run_id"],
            "run_order": record["run_order"],
            "source_file": record["source_file"],
            "started_at_utc": record["started_at_utc"],
            "initialization_ms": record["initialization_ms"],
            "first_inference_ms": record["first_inference_ms"],
        }
        for section_name in ("gpu_only", "end_to_end"):
            for metric_name, value in record[section_name].items():
                row[f"{section_name}_{metric_name}"] = value
        rows.append(row)
    return rows


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() in {"N/A", "NA", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(value)
    except ValueError:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
        return float(match.group(0)) if match else None


def load_telemetry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"telemetry_(pytorch|onnx)_cuda_b(\d+)_(.+)\.csv$",
        re.IGNORECASE,
    )
    for path in sorted(TELEMETRY_DIR.glob("telemetry_*_cuda_b*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        runtime_key = "pytorch" if match.group(1).lower() == "pytorch" else "onnxruntime"
        samples: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            samples.extend(reader)
        if not samples:
            continue

        def values(key: str) -> list[float]:
            return [
                number
                for number in (parse_number(sample.get(key)) for sample in samples)
                if number is not None
            ]

        row = {
            "runtime": RUNTIME_LABEL[runtime_key],
            "batch_size": int(match.group(2)),
            "run_id": match.group(3),
            "sample_count": len(samples),
            "gpu_utilization_mean_pct": mean_or_none(values("utilization_gpu_pct")),
            "gpu_utilization_max_pct": max(values("utilization_gpu_pct"), default=None),
            "memory_utilization_mean_pct": mean_or_none(values("utilization_memory_pct")),
            "memory_used_max_mb": max(values("memory_used_mb"), default=None),
            "temperature_max_c": max(values("temperature_gpu_c"), default=None),
            "power_mean_w": mean_or_none(values("power_draw_w")),
            "power_max_w": max(values("power_draw_w"), default=None),
            "sm_clock_mean_mhz": mean_or_none(values("clocks_sm_mhz")),
            "memory_clock_mean_mhz": mean_or_none(values("clocks_mem_mhz")),
            "source_file": path.relative_to(ROOT).as_posix(),
        }
        rows.append(row)
    return rows


def load_validations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(VALIDATION_DIR.glob("validation_cuda_b*.json")):
        data = read_json(path)
        rows.append(
            {
                "batch_size": data.get("batch_size") or filename_batch(path),
                "validation_id": data.get("validation_id"),
                "max_absolute_difference": data.get("max_absolute_difference"),
                "mean_absolute_difference": data.get("mean_absolute_difference"),
                "allclose": data.get("outputs_all_close"),
                "same_top1": data.get("same_top1"),
                "rtol": data.get("rtol"),
                "atol": data.get("atol"),
                "source_file": path.relative_to(ROOT).as_posix(),
            }
        )
    return rows


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def unique_environment(records: list[dict[str, Any]], key: str) -> str:
    values = {
        json.dumps(record["environment"].get(key), ensure_ascii=False, sort_keys=True)
        if isinstance(record["environment"].get(key), (list, dict))
        else str(record["environment"].get(key))
        for record in records
        if record["environment"].get(key) not in (None, "")
    }
    return "; ".join(sorted(values)) if values else "missing"


def generate_report(
    records: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
    validations: list[dict[str, Any]],
) -> str:
    lines = [
        "# GPU 벤치마크 보완실험 보고서",
        "",
        "## 데이터 범위",
        "",
        f"- 독립 실행 성능 JSON: {len(records)}개",
        f"- Telemetry CSV: {len(telemetry)}개",
        f"- CUDA 검증 JSON: {len(validations)}개",
        f"- Profiler 파일: {len(list(PROFILE_DIR.rglob('*.json'))) if PROFILE_DIR.exists() else 0}개",
        "",
        "## 실험 환경",
        "",
        markdown_table(
            ["항목", "기록값"],
            [
                ["호스트", unique_environment(records, "hostname")],
                ["GPU", unique_environment(records, "gpu_name")],
                ["모델", unique_environment(records, "model")],
                ["정밀도", unique_environment(records, "precision")],
                ["PyTorch", unique_environment(records, "torch")],
                ["ONNX Runtime", unique_environment(records, "onnxruntime")],
                ["CUDA Runtime", unique_environment(records, "cuda_runtime")],
                ["cuDNN", unique_environment(records, "cudnn")],
                ["NVIDIA Driver", unique_environment(records, "driver")],
                ["TF32", unique_environment(records, "tf32")],
                ["Warm-up", unique_environment(records, "warmup_count")],
                ["측정 반복", unique_environment(records, "measurement_count")],
                ["Seed", unique_environment(records, "seed")],
                ["Slurm Job ID", unique_environment(records, "slurm_job_id")],
                ["ONNX Providers", unique_environment(records, "providers")],
            ],
        ),
        "",
        "## 독립 실행 집계",
        "",
    ]

    aggregate_rows: list[list[str]] = []
    by_key = {(row["runtime_key"], row["batch_size"]): row for row in summary}
    for row in summary:
        aggregate_rows.append(
            [
                row["runtime"],
                str(row["batch_size"]),
                str(row["independent_run_count"]),
                format_value(row["gpu_only_mean_ms"]),
                format_value(row["gpu_only_run_stdev_ms"]),
                f"[{format_value(row['gpu_only_run_mean_ci95_lower_ms'])}, {format_value(row['gpu_only_run_mean_ci95_upper_ms'])}]",
                format_value(row["gpu_only_p95_mean_ms"]),
                format_value(row["gpu_only_p99_mean_ms"]),
                format_value(row["gpu_only_throughput_mean"], 2),
                format_value(row["end_to_end_mean_ms"]),
                format_value(row["end_to_end_throughput_mean"], 2),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Runtime",
                "Batch",
                "Runs",
                "GPU mean (ms)",
                "Run std (ms)",
                "Run mean 95% CI",
                "P95 mean (ms)",
                "P99 mean (ms)",
                "GPU throughput",
                "E2E mean (ms)",
                "E2E throughput",
            ],
            aggregate_rows,
        )
    )

    lines += ["", "## ONNX Runtime Speedup", ""]
    speedup_rows: list[list[str]] = []
    for batch in sorted({row["batch_size"] for row in summary}):
        pytorch = by_key.get(("pytorch", batch))
        onnx = by_key.get(("onnxruntime", batch))
        if not pytorch or not onnx:
            continue
        gpu_speedup = pytorch["gpu_only_mean_ms"] / onnx["gpu_only_mean_ms"]
        e2e_speedup = pytorch["end_to_end_mean_ms"] / onnx["end_to_end_mean_ms"]
        speedup_rows.append(
            [
                str(batch),
                format_value(gpu_speedup, 3) + "x",
                format_value(e2e_speedup, 3) + "x",
            ]
        )
    lines.append(markdown_table(["Batch", "GPU-only speedup", "E2E speedup"], speedup_rows))

    lines += ["", "## GPU Telemetry", ""]
    if telemetry:
        telemetry_rows = [
            [
                row["runtime"],
                str(row["batch_size"]),
                row["run_id"],
                str(row["sample_count"]),
                format_value(row["gpu_utilization_mean_pct"], 2),
                format_value(row["gpu_utilization_max_pct"], 2),
                format_value(row["memory_used_max_mb"], 2),
                format_value(row["temperature_max_c"], 1),
                format_value(row["power_mean_w"], 2),
                format_value(row["sm_clock_mean_mhz"], 1),
            ]
            for row in telemetry
        ]
        lines.append(
            markdown_table(
                [
                    "Runtime",
                    "Batch",
                    "Run",
                    "Samples",
                    "GPU util mean (%)",
                    "GPU util max (%)",
                    "Memory max (MB)",
                    "Temp max (C)",
                    "Power mean (W)",
                    "SM clock mean (MHz)",
                ],
                telemetry_rows,
            )
        )
    else:
        lines.append("- Telemetry 파일이 없어 GPU 상태와 성능 변동의 관계를 아직 확인할 수 없다.")

    lines += ["", "## 출력 검증", ""]
    if validations:
        validation_rows = [
            [
                str(row["batch_size"]),
                str(row["validation_id"]),
                format_value(row["max_absolute_difference"], 8),
                format_value(row["mean_absolute_difference"], 8),
                format_value(row["allclose"]),
                format_value(row["same_top1"]),
                format_value(row["rtol"]),
                format_value(row["atol"]),
            ]
            for row in validations
        ]
        lines.append(
            markdown_table(
                ["Batch", "Validation", "Max diff", "Mean diff", "Allclose", "Top-1", "rtol", "atol"],
                validation_rows,
            )
        )
    else:
        lines.append("- CUDA 출력 검증 결과가 없다.")

    incomplete_groups = [
        row for row in summary if row["independent_run_count"] < 5
    ]
    lines += ["", "## 판정", ""]
    if incomplete_groups:
        lines.append(
            "- 일부 Runtime·Batch 조합이 5회 미만이므로 독립 실행 재현성 보완은 아직 완료되지 않았다."
        )
    else:
        lines.append("- 모든 Runtime·Batch 조합에서 독립 실행 5회 이상을 확보했다.")
    lines.append(
        "- GPU-only와 End-to-End 결과는 측정 경계가 다르므로 각각 별도로 해석해야 한다."
    )
    if telemetry:
        lines.append("- Telemetry와 Latency를 같은 run_id로 연결해 부하·클럭·온도 영향을 확인할 수 있다.")
    lines.append(
        "- Profiler 결과가 존재하더라도 성능 원인은 Timeline과 Operator 자료를 직접 확인한 뒤 확정해야 한다."
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    section = normalize_section(
        {
            "mean_batch_latency_ms": 2.0,
            "median_batch_latency_ms": 1.9,
            "p95_batch_latency_ms": 2.1,
            "p99_batch_latency_ms": 2.2,
            "throughput_images_per_second": 2000.0,
        }
    )
    assert section["mean"] == 2.0
    assert filename_batch(Path("pytorch_metrics_cuda_b16_r01.json")) == 16
    lower, upper = ci95([1.0, 1.0, 1.0])
    assert lower == 1.0 and upper == 1.0


def main() -> None:
    self_check()
    records = load_metric_records()
    if not records:
        raise SystemExit(
            "No CUDA metrics found under benchmark/results/pytorch and benchmark/results/onnx"
        )
    summary = aggregate_records(records)
    telemetry = load_telemetry()
    validations = load_validations()

    write_csv(REPEATED_CSV, flatten_repeated(records))
    write_csv(SUMMARY_CSV, summary)
    write_csv(TELEMETRY_CSV, telemetry)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        generate_report(records, summary, telemetry, validations),
        encoding="utf-8",
    )
    print(f"Wrote {REPEATED_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {TELEMETRY_CSV}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        print("self-check passed")
    elif sys.argv[1:]:
        raise SystemExit("usage: python benchmark/analyze_results.py [--self-check]")
    else:
        main()
