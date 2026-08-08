#!/usr/bin/env bash
set -euo pipefail

# 목적: 대표 조건 B1/B16에만 Profiler를 적용한다.
# 성능 수치용 300회 실험과 분리해 실행한다.

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$BENCHMARK_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE_MEASUREMENTS="${PROFILE_MEASUREMENTS:-20}"
NSYS_DIR="$BENCHMARK_DIR/results/profiles/nsight"
mkdir -p "$NSYS_DIR"

for batch in 1 16; do
  "$PYTHON_BIN" "$BENCHMARK_DIR/pytorch_benchmark.py" \
    --device cuda \
    --batch "$batch" \
    --warmup 5 \
    --measurements "$PROFILE_MEASUREMENTS" \
    --run-id "profile_b${batch}" \
    --tf32 off \
    --enable-profiler \
    --profile-iterations 20

  "$PYTHON_BIN" "$BENCHMARK_DIR/onnx_benchmark.py" \
    --provider cuda \
    --batch "$batch" \
    --warmup 5 \
    --measurements "$PROFILE_MEASUREMENTS" \
    --run-id "profile_b${batch}" \
    --tf32 off \
    --disable-cpu-fallback \
    --enable-profiler
done

if command -v nsys >/dev/null 2>&1; then
  for batch in 1 16; do
    nsys profile \
      --force-overwrite=true \
      --trace=cuda,cudnn,cublas,osrt,nvtx \
      --output "$NSYS_DIR/pytorch_cuda_b${batch}" \
      "$PYTHON_BIN" "$BENCHMARK_DIR/pytorch_benchmark.py" \
        --device cuda --batch "$batch" --warmup 5 --measurements 20 \
        --run-id "nsys_b${batch}" --tf32 off

    nsys profile \
      --force-overwrite=true \
      --trace=cuda,cudnn,cublas,osrt,nvtx \
      --output "$NSYS_DIR/onnx_cuda_b${batch}" \
      "$PYTHON_BIN" "$BENCHMARK_DIR/onnx_benchmark.py" \
        --provider cuda --batch "$batch" --warmup 5 --measurements 20 \
        --run-id "nsys_b${batch}" --tf32 off --disable-cpu-fallback
  done
else
  echo "nsys command not found; built-in PyTorch/ONNX profiles were still generated."
fi

echo "GPU profiler experiment completed."
