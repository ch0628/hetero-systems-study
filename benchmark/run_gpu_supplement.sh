#!/usr/bin/env bash
set -euo pipefail

# 목적: 2 Runtime × 3 Batch를 독립 프로세스에서 반복하고,
# 각 실행과 같은 run_id로 NVIDIA telemetry를 저장한다.

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$BENCHMARK_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-10}"
MEASUREMENTS="${MEASUREMENTS:-300}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-200}"

LOG_DIR="$REPO_ROOT/results/gpu"
TELEMETRY_DIR="$BENCHMARK_DIR/results/telemetry"
mkdir -p "$LOG_DIR" "$TELEMETRY_DIR"

for batch in 1 4 16; do
  if [[ ! -f "$BENCHMARK_DIR/data/inputs/input_b${batch}.npy" ]]; then
    echo "Missing input_b${batch}.npy. Run export_onnx.py first." >&2
    exit 1
  fi
  if [[ ! -f "$BENCHMARK_DIR/data/onnx_models/resnet18_b${batch}.onnx" ]]; then
    echo "Missing resnet18_b${batch}.onnx. Run export_onnx.py first." >&2
    exit 1
  fi
done

MONITOR_PID=""
cleanup_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  MONITOR_PID=""
}
trap cleanup_monitor EXIT INT TERM

start_telemetry() {
  local output_file="$1"
  cat > "$output_file" <<'CSV'
timestamp,utilization_gpu_pct,utilization_memory_pct,memory_used_mb,memory_total_mb,temperature_gpu_c,power_draw_w,clocks_sm_mhz,clocks_mem_mhz
CSV
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader,nounits \
    -lms "$TELEMETRY_INTERVAL_MS" \
    >> "$output_file" &
  MONITOR_PID=$!
  sleep 0.3
}

conditions=(
  "pytorch:1"
  "pytorch:4"
  "pytorch:16"
  "onnx:1"
  "onnx:4"
  "onnx:16"
)

for ((repeat=1; repeat<=RUNS; repeat++)); do
  mapfile -t shuffled < <(
    "$PYTHON_BIN" - "$repeat" "$SHUFFLE_SEED" "${conditions[@]}" <<'PY'
import random
import sys

repeat = int(sys.argv[1])
seed = int(sys.argv[2])
items = sys.argv[3:]
random.Random(seed + repeat).shuffle(items)
print("\n".join(items))
PY
  )

  order=0
  for condition in "${shuffled[@]}"; do
    order=$((order + 1))
    runtime="${condition%%:*}"
    batch="${condition##*:}"
    run_id=$(printf "r%02d_o%02d" "$repeat" "$order")
    telemetry_file="$TELEMETRY_DIR/telemetry_${runtime}_cuda_b${batch}_${run_id}.csv"
    log_file="$LOG_DIR/${runtime}_cuda_b${batch}_${run_id}.txt"

    echo "============================================================"
    echo "repeat=$repeat order=$order runtime=$runtime batch=$batch run_id=$run_id"
    echo "============================================================"

    start_telemetry "$telemetry_file"
    if [[ "$runtime" == "pytorch" ]]; then
      "$PYTHON_BIN" "$BENCHMARK_DIR/pytorch_benchmark.py" \
        --device cuda \
        --batch "$batch" \
        --warmup "$WARMUP" \
        --measurements "$MEASUREMENTS" \
        --run-id "$run_id" \
        --run-order "$order" \
        --seed 42 \
        --tf32 off \
        2>&1 | tee "$log_file"
    else
      "$PYTHON_BIN" "$BENCHMARK_DIR/onnx_benchmark.py" \
        --provider cuda \
        --batch "$batch" \
        --warmup "$WARMUP" \
        --measurements "$MEASUREMENTS" \
        --run-id "$run_id" \
        --run-order "$order" \
        --seed 42 \
        --tf32 off \
        --disable-cpu-fallback \
        2>&1 | tee "$log_file"
    fi
    cleanup_monitor
    sleep 1
  done
done

# 마지막에 저장된 canonical output끼리 batch별 수치 검증
for batch in 1 4 16; do
  "$PYTHON_BIN" "$BENCHMARK_DIR/validate_outputs.py" \
    --device cuda \
    --batch "$batch" \
    --validation-id supplement
 done

"$PYTHON_BIN" "$BENCHMARK_DIR/analyze_results.py"

echo "GPU supplement experiment completed."
