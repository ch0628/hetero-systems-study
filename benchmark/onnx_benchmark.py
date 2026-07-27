import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil

from benchmark import benchmark, print_statistics


def current_rss_mb() -> float:
    """현재 프로세스가 사용 중인 실제 메모리 크기를 MB 단위로 반환한다."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def peak_rss_mb() -> float:
    """프로세스가 실행 중 기록한 최대 RSS를 MB 단위로 반환한다."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# --------------------------------------------------
# 1. 실행 옵션
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="ONNX Runtime CPU inference benchmark"
)

parser.add_argument(
    "--threads",
    type=int,
    required=True,
    help="Number of intra-op CPU threads",
)

parser.add_argument(
    "--batch",
    type=int,
    required=True,
    choices=[1, 4, 16],
    help="Batch size for inference",
)

args = parser.parse_args()


# --------------------------------------------------
# 2. 파일 및 디렉터리 경로
# --------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent

INPUT_DIR = BENCHMARK_DIR / "data" / "inputs"
MODEL_DIR = BENCHMARK_DIR / "data" / "onnx_models"
RESULT_DIR = BENCHMARK_DIR / "results" / "onnx"

# 결과 폴더만 자동 생성한다.
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = INPUT_DIR / f"input_b{args.batch}.npy"
MODEL_PATH = MODEL_DIR / f"resnet18_b{args.batch}.onnx"

OUTPUT_PATH = RESULT_DIR / f"onnx_output_b{args.batch}.npy"

METRICS_PATH = (
    RESULT_DIR
    / f"onnx_metrics_b{args.batch}_t{args.threads}.json"
)


# 입력과 모델은 export_onnx.py가 미리 생성해야 한다.
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_PATH}\n"
        f"Run: python3 export_onnx.py --batch {args.batch}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}\n"
        f"Run: python3 export_onnx.py --batch {args.batch}"
    )


# --------------------------------------------------
# 3. 입력 데이터 로딩
# --------------------------------------------------

start = time.perf_counter()

input_array = np.load(INPUT_PATH)

end = time.perf_counter()

input_load_ms = (end - start) * 1000

print(f"Input path: {INPUT_PATH}")
print(f"Input file load: {input_load_ms:.4f} ms")
print(f"Requested batch size: {args.batch}")
print(f"Actual input shape: {input_array.shape}")
print(f"Input dtype: {input_array.dtype}")


# 실행 옵션과 실제 입력의 Batch 크기가 같은지 확인한다.
if input_array.ndim != 4:
    raise ValueError(
        f"Expected a 4D input tensor, but got shape {input_array.shape}"
    )

if input_array.shape[0] != args.batch:
    raise ValueError(
        f"Input batch mismatch: "
        f"expected {args.batch}, got {input_array.shape[0]}"
    )


# --------------------------------------------------
# 4. ONNX Runtime Session 설정
# --------------------------------------------------

options = ort.SessionOptions()

options.intra_op_num_threads = args.threads
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL


# --------------------------------------------------
# 5. Session 로딩
# --------------------------------------------------

rss_before_session = current_rss_mb()

start = time.perf_counter()

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)

end = time.perf_counter()

rss_after_session = current_rss_mb()

session_load_ms = (end - start) * 1000
session_rss_delta_mb = rss_after_session - rss_before_session

print(f"Model path: {MODEL_PATH}")
print(f"RSS before session: {rss_before_session:.2f} MB")
print(f"Session RSS delta: {session_rss_delta_mb:.2f} MB")
print(f"Session load: {session_load_ms:.4f} ms")
print(f"RSS after session load: {rss_after_session:.2f} MB")


# --------------------------------------------------
# 6. 모델 입출력 정보 확인
# --------------------------------------------------

model_input = session.get_inputs()[0]
model_output = session.get_outputs()[0]

input_name = model_input.name

print(f"Model input name: {model_input.name}")
print(f"Model input shape: {model_input.shape}")
print(f"Model input type: {model_input.type}")

print(f"Model output name: {model_output.name}")
print(f"Model output shape: {model_output.shape}")
print(f"Model output type: {model_output.type}")


# --------------------------------------------------
# 7. 첫 추론
# --------------------------------------------------

start = time.perf_counter()

first_outputs = session.run(
    None,
    {input_name: input_array},
)

end = time.perf_counter()

first_inference_ms = (end - start) * 1000

# ResNet18의 첫 번째 출력 Tensor
first_output = first_outputs[0]

print(f"First inference: {first_inference_ms:.4f} ms")
print(f"Actual output shape: {first_output.shape}")


# 정확성 검증에 사용할 출력 저장
np.save(
    OUTPUT_PATH,
    first_output,
)

print(f"ONNX output saved: {OUTPUT_PATH}")


# --------------------------------------------------
# 8. 반복 추론 함수
# --------------------------------------------------

def inference():
    return session.run(
        None,
        {input_name: input_array},
    )


print(f"Provider: {session.get_providers()}")
print(
    f"ONNX Runtime intra-op threads: "
    f"{options.intra_op_num_threads}"
)
print(
    f"ONNX Runtime inter-op threads: "
    f"{options.inter_op_num_threads}"
)
print(f"Execution mode: {options.execution_mode}")


# --------------------------------------------------
# 9. Warm 추론 Benchmark
# --------------------------------------------------

times = benchmark(
    inference_fn=inference,
    warming_count=10,
    measured_count=300,
)

print_statistics(times)


# --------------------------------------------------
# 10. Batch 관련 성능 지표
# --------------------------------------------------

mean_batch_latency_ms = float(np.mean(times))
median_batch_latency_ms = float(np.median(times))
p95_batch_latency_ms = float(np.percentile(times, 95))
min_batch_latency_ms = float(min(times))
max_batch_latency_ms = float(max(times))

# Batch 전체 지연시간을 이미지 개수로 나눈 값
per_image_latency_ms = (
    mean_batch_latency_ms / args.batch
)

# 초당 처리할 수 있는 이미지 개수
throughput_images_per_second = (
    args.batch / (mean_batch_latency_ms / 1000)
)

final_rss_mb = current_rss_mb()
peak_memory_mb = peak_rss_mb()

print(f"Batch size: {args.batch}")
print(
    f"Mean batch latency: "
    f"{mean_batch_latency_ms:.4f} ms"
)
print(
    f"Per-image latency: "
    f"{per_image_latency_ms:.4f} ms/image"
)
print(
    f"Throughput: "
    f"{throughput_images_per_second:.2f} images/s"
)
print(f"Final RSS: {final_rss_mb:.2f} MB")
print(f"Peak RSS: {peak_memory_mb:.2f} MB")


# --------------------------------------------------
# 11. 성능 측정 결과 JSON 저장
# --------------------------------------------------

metrics = {
    "runtime": "onnxruntime",
    "provider": "CPUExecutionProvider",
    "model": "resnet18",
    "model_path": str(MODEL_PATH),
    "input_path": str(INPUT_PATH),
    "output_path": str(OUTPUT_PATH),
    "batch_size": args.batch,
    "input_shape": list(input_array.shape),
    "output_shape": list(first_output.shape),
    "input_dtype": str(input_array.dtype),
    "intra_op_threads": args.threads,
    "inter_op_threads": 1,
    "execution_mode": "ORT_SEQUENTIAL",
    "warmup_count": 10,
    "measurement_count": 300,
    "input_load_ms": input_load_ms,
    "session_load_ms": session_load_ms,
    "rss_before_session_mb": rss_before_session,
    "rss_after_session_mb": rss_after_session,
    "session_rss_delta_mb": session_rss_delta_mb,
    "first_inference_ms": first_inference_ms,
    "mean_batch_latency_ms": mean_batch_latency_ms,
    "median_batch_latency_ms": median_batch_latency_ms,
    "p95_batch_latency_ms": p95_batch_latency_ms,
    "min_batch_latency_ms": min_batch_latency_ms,
    "max_batch_latency_ms": max_batch_latency_ms,
    "per_image_latency_ms": per_image_latency_ms,
    "throughput_images_per_second": (
        throughput_images_per_second
    ),
    "final_rss_mb": final_rss_mb,
    "peak_rss_mb": peak_memory_mb,
}

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

print(f"Metrics saved: {METRICS_PATH}")