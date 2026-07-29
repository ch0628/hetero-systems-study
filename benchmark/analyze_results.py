from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


MISSING = "missing"
METRICS = ("mean", "median", "p95", "min", "max")
RUNTIMES = ("PyTorch CUDA", "ONNX Runtime CUDA")

ROOT = Path(__file__).resolve().parents[1]
GPU_DIR = ROOT / "results" / "gpu"
VALIDATION_DIR = ROOT / "benchmark" / "results" / "validation"
CSV_PATH = ROOT / "benchmark" / "results" / "gpu_summary.csv"
REPORT_PATH = ROOT / "docs" / "gpu_benchmark_report.md"

CSV_FIELDS = [
    "runtime",
    "batch_size",
    "source_file",
    "gpu_only_mean_ms",
    "gpu_only_median_ms",
    "gpu_only_p95_ms",
    "gpu_only_min_ms",
    "gpu_only_max_ms",
    "end_to_end_mean_ms",
    "end_to_end_median_ms",
    "end_to_end_p95_ms",
    "end_to_end_min_ms",
    "end_to_end_max_ms",
    "first_inference_ms",
    "initialization_type",
    "initialization_ms",
    "gpu_only_throughput_samples_per_s",
    "end_to_end_throughput_samples_per_s",
    "onnx_gpu_only_speedup",
    "onnx_end_to_end_speedup",
    "validation_cuda_max_difference",
    "validation_cuda_mean_difference",
    "validation_cuda_allclose",
    "validation_cuda_top1_match",
    "validation_unspecified_device_max_difference",
    "validation_unspecified_device_mean_difference",
    "validation_unspecified_device_allclose",
    "validation_unspecified_device_top1_match",
]


def extract_batch(filename: str) -> int | None:
    match = re.search(r"(?:^|[_-])b(?:atch)?[_-]?(\d+)(?:$|[_-])", Path(filename).stem.lower())
    return int(match.group(1)) if match else None


def extract_runtime(filename: str) -> str:
    stem = Path(filename).stem.lower()
    if "pytorch" in stem and "cuda" in stem:
        return "PyTorch CUDA"
    if ("onnx" in stem or "onnxruntime" in stem) and "cuda" in stem:
        return "ONNX Runtime CUDA"
    return MISSING


def number_ms(line: str) -> float | None:
    match = re.search(
        r":\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(ms|s|us|µs)?\s*$",
        line,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = (match.group(2) or "ms").lower()
    return value * {"s": 1000.0, "ms": 1.0, "us": 0.001, "µs": 0.001}[unit]


def labeled_ms(lines: list[str], label: str) -> float | None:
    pattern = re.compile(rf"^\s*{re.escape(label)}\s*:", re.IGNORECASE)
    for line in lines:
        if pattern.match(line):
            return number_ms(line)
    return None


def labeled_text(lines: list[str], label: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(label)}\s*:\s*(.*?)\s*$", re.IGNORECASE)
    for line in lines:
        match = pattern.match(line)
        if match:
            return match.group(1) or None
    return None


def parse_latency_sections(lines: list[str]) -> dict[str, dict[str, float | None]]:
    sections = {
        "gpu_only": {metric: None for metric in METRICS},
        "end_to_end": {metric: None for metric in METRICS},
    }
    current: str | None = None
    for line in lines:
        lowered = line.lower()
        if line.strip().startswith("==="):
            if "gpu-only" in lowered or "gpu only" in lowered:
                current = "gpu_only"
            elif "end-to-end" in lowered or "end to end" in lowered:
                current = "end_to_end"
            else:
                current = None
            continue
        if current is None:
            continue
        match = re.match(r"^\s*(mean|median|p95|min|max)\s*:", line, re.IGNORECASE)
        if match:
            sections[current][match.group(1).lower()] = number_ms(line)
    return sections


def parse_gpu_result(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    runtime = extract_runtime(path.name)
    batch = extract_batch(path.name)
    sections = parse_latency_sections(lines)
    initialization_label = "Model initialization" if runtime == "PyTorch CUDA" else "Session loading"
    return {
        "runtime": runtime,
        "batch_size": batch,
        "source_file": path.relative_to(ROOT).as_posix(),
        "gpu_only": sections["gpu_only"],
        "end_to_end": sections["end_to_end"],
        "first_inference_ms": labeled_ms(lines, "First inference"),
        "initialization_type": initialization_label,
        "initialization_ms": labeled_ms(lines, initialization_label),
        "environment": {
            "hostname": labeled_text(lines, "Hostname"),
            "device": labeled_text(lines, "Device"),
            "gpu": labeled_text(lines, "GPU"),
            "pytorch": labeled_text(lines, "PyTorch"),
            "cuda_runtime": labeled_text(lines, "CUDA runtime"),
            "cudnn": labeled_text(lines, "cuDNN"),
            "provider_request": labeled_text(lines, "Provider request"),
            "actual_providers": labeled_text(lines, "Actual providers"),
            "input_shape": labeled_text(lines, "Input shape"),
            "output_shape": labeled_text(lines, "Output shape"),
        },
    }


def parse_validation(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    filename_batch = extract_batch(path.name)
    batch = data.get("batch_size", filename_batch)
    profile = "cuda" if data.get("device") == "cuda" or "_cuda_" in path.stem.lower() else "unspecified_device"
    return {
        "profile": profile,
        "batch_size": batch,
        "source_file": path.relative_to(ROOT).as_posix(),
        "max_difference": data.get("max_absolute_difference"),
        "mean_difference": data.get("mean_absolute_difference"),
        "allclose": data.get("outputs_all_close"),
        "top1_match": data.get("same_top1"),
        "rtol": data.get("rtol"),
        "atol": data.get("atol"),
        "output_shape": data.get("output_shape"),
    }


def divide(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def throughput(batch: int | None, mean_latency_ms: float | None) -> float | None:
    ratio = divide(batch, mean_latency_ms)
    return ratio * 1000.0 if ratio is not None else None


def text_value(value: Any, digits: int | None = None) -> str:
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.{digits}f}" if digits is not None else f"{value:.15g}"
    return str(value)


def environment_values(records: list[dict[str, Any]], key: str) -> str:
    values = sorted({record["environment"].get(key) for record in records if record["environment"].get(key)})
    return "; ".join(values) if values else MISSING


def build_rows(
    records: list[dict[str, Any]], validations: list[dict[str, Any]]
) -> tuple[list[dict[str, str]], dict[tuple[str, int], dict[str, Any]]]:
    by_runtime_batch = {
        (record["runtime"], record["batch_size"]): record
        for record in records
        if record["runtime"] in RUNTIMES and record["batch_size"] is not None
    }
    validations_by_profile_batch = {
        (validation["profile"], validation["batch_size"]): validation
        for validation in validations
        if validation["batch_size"] is not None
    }
    rows: list[dict[str, str]] = []
    for runtime, batch in sorted(
        by_runtime_batch,
        key=lambda item: (item[1], RUNTIMES.index(item[0])),
    ):
        record = by_runtime_batch[(runtime, batch)]
        pytorch = by_runtime_batch.get(("PyTorch CUDA", batch))
        onnx = by_runtime_batch.get(("ONNX Runtime CUDA", batch))
        gpu_speedup = (
            divide(pytorch["gpu_only"]["mean"], onnx["gpu_only"]["mean"])
            if runtime == "ONNX Runtime CUDA" and pytorch and onnx
            else None
        )
        e2e_speedup = (
            divide(pytorch["end_to_end"]["mean"], onnx["end_to_end"]["mean"])
            if runtime == "ONNX Runtime CUDA" and pytorch and onnx
            else None
        )
        row: dict[str, str] = {
            "runtime": runtime,
            "batch_size": str(batch),
            "source_file": record["source_file"],
            "first_inference_ms": text_value(record["first_inference_ms"], 4),
            "initialization_type": record["initialization_type"],
            "initialization_ms": text_value(record["initialization_ms"], 4),
            "gpu_only_throughput_samples_per_s": text_value(
                throughput(batch, record["gpu_only"]["mean"]), 6
            ),
            "end_to_end_throughput_samples_per_s": text_value(
                throughput(batch, record["end_to_end"]["mean"]), 6
            ),
            "onnx_gpu_only_speedup": text_value(gpu_speedup, 6),
            "onnx_end_to_end_speedup": text_value(e2e_speedup, 6),
        }
        for section, prefix in (("gpu_only", "gpu_only"), ("end_to_end", "end_to_end")):
            for metric in METRICS:
                row[f"{prefix}_{metric}_ms"] = text_value(record[section][metric], 4)
        for profile in ("cuda", "unspecified_device"):
            validation = validations_by_profile_batch.get((profile, batch), {})
            prefix = f"validation_{profile}"
            row[f"{prefix}_max_difference"] = text_value(validation.get("max_difference"))
            row[f"{prefix}_mean_difference"] = text_value(validation.get("mean_difference"))
            row[f"{prefix}_allclose"] = text_value(validation.get("allclose"))
            row[f"{prefix}_top1_match"] = text_value(validation.get("top1_match"))
        rows.append({field: row.get(field, MISSING) for field in CSV_FIELDS})
    return rows, by_runtime_batch


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def metric_value(record: dict[str, Any] | None, section: str, metric: str = "mean") -> float | None:
    return record[section][metric] if record else None


def pct_change(current: float | None, baseline: float | None) -> float | None:
    ratio = divide(current, baseline)
    return (ratio - 1.0) * 100.0 if ratio is not None else None


def report_text(
    records: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    by_runtime_batch: dict[tuple[str, int], dict[str, Any]],
) -> str:
    batches = sorted({batch for _, batch in by_runtime_batch})
    lines = [
        "# GPU 벤치마크 분석 보고서",
        "",
        "## 데이터 범위",
        "",
        f"- 성능 로그: `results/gpu/*.txt` ({len(records)}개)",
        f"- 검증 결과: `benchmark/results/validation/*.json` ({len(validations)}개)",
        "- 원본 파일은 읽기만 했으며 수정하지 않았다.",
        "- 값이 기록되지 않은 항목은 `missing`으로 표기했다.",
        "",
        "## 실험 환경",
        "",
    ]
    lines += markdown_table(
        ["항목", "기록값"],
        [
            ["호스트", environment_values(records, "hostname")],
            ["GPU", environment_values(records, "gpu")],
            ["요청 장치/Provider", environment_values(records, "device") + "; " + environment_values(records, "provider_request")],
            ["ONNX 실제 Provider", environment_values(records, "actual_providers")],
            ["PyTorch", environment_values(records, "pytorch")],
            ["ONNX Runtime 버전", MISSING],
            ["CUDA runtime", environment_values(records, "cuda_runtime")],
            ["cuDNN", environment_values(records, "cudnn")],
            ["입력 shape", environment_values(records, "input_shape")],
            ["출력 shape", environment_values(records, "output_shape")],
            ["모델명", MISSING],
            ["정밀도", MISSING],
            ["warm-up/측정 반복 수", MISSING],
        ],
    )
    lines += [
        "",
        "환경 항목은 로그에 명시된 값만 사용했다. `Actual providers`에 CPU fallback Provider도 함께 등록되어 있지만, 연산별 Provider 배치는 로그에 없어 확인할 수 없다.",
        "",
        "## 전체 측정 결과",
        "",
        "### GPU-only latency",
        "",
    ]
    gpu_rows: list[list[str]] = []
    e2e_rows: list[list[str]] = []
    startup_rows: list[list[str]] = []
    for batch in batches:
        for runtime in RUNTIMES:
            record = by_runtime_batch.get((runtime, batch))
            if not record:
                continue
            gpu_rows.append(
                [runtime, str(batch)]
                + [text_value(record["gpu_only"][metric], 4) for metric in METRICS]
                + [text_value(throughput(batch, record["gpu_only"]["mean"]), 2)]
            )
            e2e_rows.append(
                [runtime, str(batch)]
                + [text_value(record["end_to_end"][metric], 4) for metric in METRICS]
                + [text_value(throughput(batch, record["end_to_end"]["mean"]), 2)]
            )
            startup_rows.append(
                [
                    runtime,
                    str(batch),
                    record["initialization_type"],
                    text_value(record["initialization_ms"], 4),
                    text_value(record["first_inference_ms"], 4),
                ]
            )
    lines += markdown_table(
        ["Runtime", "Batch", "Mean (ms)", "Median (ms)", "P95 (ms)", "Min (ms)", "Max (ms)", "Throughput (samples/s)"],
        gpu_rows,
    )
    lines += ["", "### End-to-End latency", ""]
    lines += markdown_table(
        ["Runtime", "Batch", "Mean (ms)", "Median (ms)", "P95 (ms)", "Min (ms)", "Max (ms)", "Throughput (samples/s)"],
        e2e_rows,
    )
    lines += ["", "### 초기화와 first inference", ""]
    lines += markdown_table(
        ["Runtime", "Batch", "초기화 종류", "초기화 (ms)", "First inference (ms)"],
        startup_rows,
    )
    lines += [
        "",
        "Throughput은 각 구간의 mean latency에 대해 `batch / (mean_latency_ms / 1000)`으로 계산했다. 초기화와 first inference는 steady-state latency/throughput 계산에서 제외했다.",
        "",
        "## PyTorch와 ONNX Runtime 비교",
        "",
    ]
    comparison_rows: list[list[str]] = []
    for batch in batches:
        pytorch = by_runtime_batch.get(("PyTorch CUDA", batch))
        onnx = by_runtime_batch.get(("ONNX Runtime CUDA", batch))
        comparison_rows.append(
            [
                str(batch),
                text_value(metric_value(pytorch, "gpu_only"), 4),
                text_value(metric_value(onnx, "gpu_only"), 4),
                text_value(divide(metric_value(pytorch, "gpu_only"), metric_value(onnx, "gpu_only")), 3),
                text_value(metric_value(pytorch, "end_to_end"), 4),
                text_value(metric_value(onnx, "end_to_end"), 4),
                text_value(divide(metric_value(pytorch, "end_to_end"), metric_value(onnx, "end_to_end")), 3),
            ]
        )
    lines += markdown_table(
        [
            "Batch",
            "PyTorch GPU mean (ms)",
            "ONNX GPU mean (ms)",
            "ONNX GPU speedup",
            "PyTorch E2E mean (ms)",
            "ONNX E2E mean (ms)",
            "ONNX E2E speedup",
        ],
        comparison_rows,
    )
    gpu_speedups = [
        divide(
            metric_value(by_runtime_batch.get(("PyTorch CUDA", batch)), "gpu_only"),
            metric_value(by_runtime_batch.get(("ONNX Runtime CUDA", batch)), "gpu_only"),
        )
        for batch in batches
    ]
    e2e_speedups = [
        divide(
            metric_value(by_runtime_batch.get(("PyTorch CUDA", batch)), "end_to_end"),
            metric_value(by_runtime_batch.get(("ONNX Runtime CUDA", batch)), "end_to_end"),
        )
        for batch in batches
    ]
    lines += [
        "",
        f"- ONNX Runtime이 모든 batch에서 더 낮은 mean latency를 기록했다. GPU-only speedup 범위: {text_value(min(x for x in gpu_speedups if x is not None), 3)}x–{text_value(max(x for x in gpu_speedups if x is not None), 3)}x.",
        f"- End-to-End speedup 범위: {text_value(min(x for x in e2e_speedups if x is not None), 3)}x–{text_value(max(x for x in e2e_speedups if x is not None), 3)}x.",
        "- batch가 커질수록 ONNX Runtime의 우위가 감소했다. 이는 측정된 경향이며 원인은 이 로그만으로 확정할 수 없다.",
        "",
        "## Batch 1, 4, 16 확장성",
        "",
    ]
    scaling_rows: list[list[str]] = []
    for runtime in RUNTIMES:
        base = by_runtime_batch.get((runtime, 1))
        base_gpu_latency = metric_value(base, "gpu_only")
        base_e2e_latency = metric_value(base, "end_to_end")
        base_gpu_throughput = throughput(1, base_gpu_latency)
        base_e2e_throughput = throughput(1, base_e2e_latency)
        for batch in batches:
            record = by_runtime_batch.get((runtime, batch))
            gpu_latency = metric_value(record, "gpu_only")
            e2e_latency = metric_value(record, "end_to_end")
            gpu_tp = throughput(batch, gpu_latency)
            e2e_tp = throughput(batch, e2e_latency)
            scaling_rows.append(
                [
                    runtime,
                    str(batch),
                    text_value(gpu_latency, 4),
                    text_value(pct_change(gpu_latency, base_gpu_latency), 2) + ("%" if gpu_latency is not None and base_gpu_latency is not None else ""),
                    text_value(gpu_tp, 2),
                    text_value(divide(gpu_tp, base_gpu_throughput), 2) + ("x" if gpu_tp is not None and base_gpu_throughput is not None else ""),
                    text_value(e2e_latency, 4),
                    text_value(e2e_tp, 2),
                    text_value(divide(e2e_tp, base_e2e_throughput), 2) + ("x" if e2e_tp is not None and base_e2e_throughput is not None else ""),
                ]
            )
    lines += markdown_table(
        [
            "Runtime",
            "Batch",
            "GPU mean (ms)",
            "GPU latency vs B1",
            "GPU throughput",
            "GPU throughput vs B1",
            "E2E mean (ms)",
            "E2E throughput",
            "E2E throughput vs B1",
        ],
        scaling_rows,
    )
    lines += ["", "## Latency와 throughput 분석", ""]
    for runtime in RUNTIMES:
        b1 = by_runtime_batch.get((runtime, 1))
        b16 = by_runtime_batch.get((runtime, 16))
        lines.append(
            f"- {runtime}: batch 1→16에서 GPU-only mean latency는 "
            f"{text_value(metric_value(b1, 'gpu_only'), 4)} ms→{text_value(metric_value(b16, 'gpu_only'), 4)} ms, "
            f"throughput은 {text_value(throughput(1, metric_value(b1, 'gpu_only')), 2)}→"
            f"{text_value(throughput(16, metric_value(b16, 'gpu_only')), 2)} samples/s였다."
        )
    for batch in batches:
        for runtime in RUNTIMES:
            record = by_runtime_batch.get((runtime, batch))
            gpu_mean = metric_value(record, "gpu_only")
            e2e_mean = metric_value(record, "end_to_end")
            overhead = None if gpu_mean is None or e2e_mean is None else e2e_mean - gpu_mean
            overhead_pct = None if overhead is None else divide(overhead, gpu_mean)
            lines.append(
                f"- {runtime}, batch {batch}: E2E와 GPU-only mean 차이는 "
                f"{text_value(overhead, 4)} ms ({text_value(None if overhead_pct is None else overhead_pct * 100, 2)}%)."
            )
    lines += [
        "",
        "GPU-only는 장치 실행 중심, End-to-End는 호출·전송·출력 처리 오버헤드를 포함한 지표다. 따라서 두 throughput은 서로 다른 운영 경계를 나타낸다.",
        "",
        "## Validation 분석",
        "",
    ]
    validation_rows = [
        [
            validation["profile"],
            str(validation["batch_size"]) if validation["batch_size"] is not None else MISSING,
            text_value(validation["max_difference"]),
            text_value(validation["mean_difference"]),
            text_value(validation["allclose"]),
            text_value(validation["top1_match"]),
            text_value(validation["rtol"]),
            text_value(validation["atol"]),
            validation["source_file"],
        ]
        for validation in sorted(validations, key=lambda item: (item["profile"], item["batch_size"] or -1))
    ]
    lines += markdown_table(
        ["Profile", "Batch", "Max difference", "Mean difference", "Allclose", "Top-1 일치", "rtol", "atol", "Source"],
        validation_rows,
    )
    cuda_validations = [item for item in validations if item["profile"] == "cuda"]
    unspecified_validations = [item for item in validations if item["profile"] == "unspecified_device"]
    lines += [
        "",
        f"- CUDA profile: allclose 통과 {sum(item['allclose'] is True for item in cuda_validations)}/{len(cuda_validations)}, Top-1 일치 {sum(item['top1_match'] is True for item in cuda_validations)}/{len(cuda_validations)}.",
        f"- 장치 미기록 profile: allclose 통과 {sum(item['allclose'] is True for item in unspecified_validations)}/{len(unspecified_validations)}, Top-1 일치 {sum(item['top1_match'] is True for item in unspecified_validations)}/{len(unspecified_validations)}.",
        "- CUDA profile은 모든 batch에서 설정된 `rtol=0.0001`, `atol=1e-05` 기준 allclose에 실패했지만 Top-1 class는 모두 일치했다.",
        "- 장치 미기록 profile은 모든 batch에서 allclose와 Top-1 일치가 모두 true였다. 해당 JSON에 장치 정보가 없어 CPU 결과라고 단정하지 않았다.",
        "",
        "## 이상치와 가능한 원인",
        "",
    ]
    outliers: list[str] = []
    for record in records:
        for section, label in (("gpu_only", "GPU-only"), ("end_to_end", "End-to-End")):
            p95 = record[section]["p95"]
            maximum = record[section]["max"]
            ratio = divide(maximum, p95)
            if ratio is not None and ratio > 1.2:
                outliers.append(
                    f"- {record['runtime']}, batch {record['batch_size']}, {label}: Max {text_value(maximum, 4)} ms가 P95 {text_value(p95, 4)} ms의 {text_value(ratio, 2)}배. 이상치 후보."
                )
    lines += outliers or ["- `Max > 1.2 × P95` 기준 이상치 후보 없음."]
    lines += [
        "- 가능한 원인: CUDA 비동기 동기화 지점, 최초 kernel/context 준비, 메모리 할당·복사, OS scheduling, 다른 프로세스의 GPU 점유, 온도·클럭 변동. 로그만으로 어느 원인인지 확정할 수 없다.",
        "- first inference와 initialization이 batch/실행별로 크게 달랐다. 실행 순서, 프로세스 재사용, cache 상태가 기록되지 않아 직접 비교에 주의가 필요하다.",
        "- CUDA validation 차이의 가능한 원인에는 backend별 연산 순서, kernel 구현, TF32/FP32 처리, 누적 반올림이 있다. 정밀도 설정과 연산별 오차 자료가 없어 확정할 수 없다.",
        "",
        "## 실험 한계",
        "",
        "- 모델명, ONNX Runtime 버전, 정밀도 설정, warm-up 횟수, 측정 반복 수가 로그에 없다.",
        "- 원시 iteration latency가 없어 분포 모양, 표준편차, confidence interval을 재계산할 수 없다.",
        "- 단일 호스트와 단일 GPU 결과라 다른 GPU·드라이버·소프트웨어 조합으로 일반화할 수 없다.",
        "- 실행 순서, 독립 프로세스 여부, cache 초기화, GPU clock·temperature·utilization이 기록되지 않았다.",
        "- `Actual providers`는 등록 Provider만 보여 주며 각 ONNX node가 CUDA에서 실행됐는지 증명하지 않는다.",
        "- validation 두 profile 중 장치 미기록 파일은 실행 장치를 확정할 수 없다.",
        "",
        "## 다음 실험 제안",
        "",
        "1. 모델명, ONNX Runtime/driver 버전, precision, warm-up·반복 수, seed를 로그에 추가한다.",
        "2. runtime·batch 조합을 독립 프로세스에서 여러 번 무작위 순서로 실행하고 원시 iteration latency를 저장한다.",
        "3. mean/median/P95뿐 아니라 표준편차와 95% confidence interval을 계산한다.",
        "4. GPU utilization, clock, temperature, power, memory 사용량을 측정 구간과 함께 기록한다.",
        "5. ONNX Runtime profiling으로 node별 Execution Provider 배치를 확인한다.",
        "6. TF32 허용 여부와 FP32/FP16 설정을 고정해 CUDA allclose 실패 원인을 분리한다.",
        "7. batch 2, 8, 32 및 saturation/OOM 지점까지 확장해 throughput 포화 구간을 찾는다.",
        "8. first inference를 CUDA context 준비, model/session 초기화, memory allocation, kernel 준비 단계로 분리 측정한다.",
        "",
    ]
    return "\n".join(lines)


def self_check() -> None:
    assert extract_runtime("pytorch_cuda_b16.txt") == "PyTorch CUDA"
    assert extract_runtime("onnx_cuda_b4.txt") == "ONNX Runtime CUDA"
    assert extract_batch("onnx_cuda_b4.txt") == 4
    sample = [
        "=== CUDA GPU-only inference ===",
        "Mean: 2.0000 ms",
        "Median: 1.9000 ms",
        "P95: 2.1000 ms",
        "Min: 1.8000 ms",
        "Max: 2.2000 ms",
        "=== CUDA End-to-End inference ===",
        "Mean: 2.5000 ms",
    ]
    parsed = parse_latency_sections(sample)
    assert parsed["gpu_only"]["mean"] == 2.0
    assert parsed["end_to_end"]["mean"] == 2.5
    assert throughput(4, 2.0) == 2000.0
    assert divide(4.0, 2.0) == 2.0


def main() -> None:
    self_check()
    records = [parse_gpu_result(path) for path in sorted(GPU_DIR.glob("*.txt"))]
    validations = [parse_validation(path) for path in sorted(VALIDATION_DIR.glob("*.json"))]
    rows, by_runtime_batch = build_rows(records, validations)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        report_text(records, validations, by_runtime_batch),
        encoding="utf-8",
    )
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        print("self-check passed")
    elif sys.argv[1:]:
        raise SystemExit("usage: python benchmark/analyze_results.py [--self-check]")
    else:
        main()
