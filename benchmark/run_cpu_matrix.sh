#!/usr/bin/env bash
set -euo pipefail

BATCHES=(1 4 16)
THREADS=(1 2 4)

# Fixed-shape ONNX models and export metadata are created once per Batch.
for batch in "${BATCHES[@]}"; do
  python3 export_onnx.py --batch "$batch" --opset 17
done

# Each command runs in an independent Python process.
for threads in "${THREADS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    python3 pytorch_benchmark.py \
      --threads "$threads" \
      --batch "$batch" \
      --warmup 10 \
      --iterations 300

    python3 onnx_benchmark.py \
      --threads "$threads" \
      --batch "$batch" \
      --warmup 10 \
      --iterations 300

    python3 validate_outputs.py \
      --threads "$threads" \
      --batch "$batch"
  done
done

python3 analyze_cpu_results.py --self-check
python3 analyze_cpu_results.py
