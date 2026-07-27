import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torchvision.models import resnet18, ResNet18_Weights

from benchmark import benchmark, print_statistics


def current_rss_mb() -> float:
    """현재 프로세스 RSS를 MB 단위로 반환한다."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def peak_rss_mb() -> float:
    """프로세스 실행 중 최대 RSS를 MB 단위로 반환한다."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# --------------------------------------------------
# 1. 실행 옵션
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="PyTorch CPU inference benchmark"
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
RESULT_DIR = BENCHMARK_DIR / "results" / "pytorch"

# 결과 저장 폴더만 자동 생성한다.
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = INPUT_DIR / f"input_b{args.batch}.npy"

PYTORCH_OUTPUT_PATH = (
    RESULT_DIR
    / f"pytorch_output_b{args.batch}.npy"
)

PYTORCH_METRICS_PATH = (
    RESULT_DIR
    / f"pytorch_metrics_b{args.batch}_t{args.threads}.json"
)


# 입력 파일은 export_onnx.py가 미리 생성해야 한다.
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_PATH}\n"
        f"Run: python3 export_onnx.py --batch {args.batch}"
    )


# --------------------------------------------------
# 3. CPU 스레드 설정
# --------------------------------------------------

torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)


# --------------------------------------------------
# 4. 모델 준비
# --------------------------------------------------

rss_before_model = current_rss_mb()

start = time.perf_counter()

model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)
model.eval()

end = time.perf_counter()

rss_after_model = current_rss_mb()

model_initialization_ms = (end - start) * 1000
model_rss_delta_mb = rss_after_model - rss_before_model

print(f"RSS before model: {rss_before_model:.2f} MB")
print(f"Model RSS delta: {model_rss_delta_mb:.2f} MB")
print(
    f"Model initialization time: "
    f"{model_initialization_ms:.4f} ms"
)
print(
    f"RSS after model load: "
    f"{rss_after_model:.2f} MB"
)


# --------------------------------------------------
# 5. 입력 데이터 로딩
# --------------------------------------------------

start = time.perf_counter()

input_array = np.load(INPUT_PATH)

end = time.perf_counter()

input_load_ms = (end - start) * 1000

print(f"Input path: {INPUT_PATH}")
print(f"Input loading time: {input_load_ms:.4f} ms")
print(f"Requested batch size: {args.batch}")
print(f"Actual input shape: {input_array.shape}")
print(f"Input dtype: {input_array.dtype}")


# ResNet18 입력은 4차원이어야 한다.
if input_array.ndim != 4:
    raise ValueError(
        f"Expected a 4D input tensor, "
        f"but got shape {input_array.shape}"
    )

# 실행 옵션과 실제 Batch 크기를 비교한다.
if input_array.shape[0] != args.batch:
    raise ValueError(
        f"Input batch mismatch: "
        f"expected {args.batch}, "
        f"got {input_array.shape[0]}"
    )


# --------------------------------------------------
# 6. NumPy 배열을 PyTorch Tensor로 변환
# --------------------------------------------------

start = time.perf_counter()

input_tensor = torch.from_numpy(input_array)

end = time.perf_counter()

tensor_conversion_ms = (end - start) * 1000

print(
    f"Input tensor conversion time: "
    f"{tensor_conversion_ms:.4f} ms"
)


# --------------------------------------------------
# 7. 첫 추론
# --------------------------------------------------

with torch.inference_mode():
    start = time.perf_counter()

    first_output = model(input_tensor)

    end = time.perf_counter()

first_inference_ms = (end - start) * 1000

print(f"First inference: {first_inference_ms:.4f} ms")
print(f"Output shape: {tuple(first_output.shape)}")


# 정확성 비교에 사용할 출력 저장
pytorch_output_array = (
    first_output
    .detach()
    .cpu()
    .numpy()
)

np.save(
    PYTORCH_OUTPUT_PATH,
    pytorch_output_array,
)

print(
    f"PyTorch output saved: "
    f"{PYTORCH_OUTPUT_PATH}"
)


# --------------------------------------------------
# 8. 반복 추론 함수
# --------------------------------------------------

def inference():
    with torch.inference_mode():
        return model(input_tensor)


print(
    "PyTorch intra-op threads:",
    torch.get_num_threads(),
)

print(
    "PyTorch inter-op threads:",
    torch.get_num_interop_threads(),
)


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

# Batch 전체 추론시간을 이미지 개수로 나눈 값
per_image_latency_ms = (
    mean_batch_latency_ms / args.batch
)

# 1초 동안 처리할 수 있는 이미지 수
throughput_images_per_second = (
    args.batch
    / (mean_batch_latency_ms / 1000)
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
# 11. 성능 결과 JSON 저장
# --------------------------------------------------

metrics = {
    "runtime": "pytorch",
    "device": "cpu",
    "model": "resnet18",
    "input_path": str(INPUT_PATH),
    "output_path": str(PYTORCH_OUTPUT_PATH),
    "batch_size": args.batch,
    "input_shape": list(input_array.shape),
    "output_shape": list(pytorch_output_array.shape),
    "input_dtype": str(input_array.dtype),
    "intra_op_threads": torch.get_num_threads(),
    "inter_op_threads": torch.get_num_interop_threads(),
    "warmup_count": 10,
    "measurement_count": 300,
    "input_load_ms": input_load_ms,
    "tensor_conversion_ms": tensor_conversion_ms,
    "model_initialization_ms": model_initialization_ms,
    "rss_before_model_mb": rss_before_model,
    "rss_after_model_mb": rss_after_model,
    "model_rss_delta_mb": model_rss_delta_mb,
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

with PYTORCH_METRICS_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metrics,
        file,
        indent=2,
        ensure_ascii=False,
    )

print(
    f"Metrics saved: "
    f"{PYTORCH_METRICS_PATH}"
)