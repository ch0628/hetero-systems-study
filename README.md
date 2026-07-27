# Heterogeneous Systems Study

PyTorch와 ONNX Runtime의 CPU 추론 성능을 비교하고,
이후 GPU, ARM, 네트워크 환경으로 확장하기 위한 벤치마크 프로젝트입니다.

## Current Experiment

- Model: ResNet18
- Device: CPU
- Runtime: PyTorch, ONNX Runtime
- Threads: 1, 2, 4
- Batch sizes: 1, 4, 16
- Warm-up: 10
- Measurements: 300

## Run

```bash
cd benchmark

python3 export_onnx.py --batch 1
python3 onnx_benchmark.py --threads 4 --batch 1
python3 pytorch_benchmark.py --threads 4 --batch 1
python3 validate_outputs.py --batch 1
