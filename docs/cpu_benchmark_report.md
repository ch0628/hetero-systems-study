# CPU Benchmark Report

## 데이터 범위

- PyTorch 결과 3개: `benchmark/results/pytorch/*.json`
- ONNX Runtime 결과 3개: `benchmark/results/onnx/*.json`
- Validation 결과 3개: `benchmark/results/validation/*.json`
- 분석 Batch: 1, 4, 16. 원본 JSON은 읽기만 했다.

## 실험 환경

| 항목 | 기록값 |
| --- | --- |
| Model | resnet18 |
| Runtime | PyTorch CPU, ONNX Runtime CPU (CPUExecutionProvider) |
| Intra-op threads | 4 |
| Inter-op threads | 1 |
| Warm-up | 10 |
| 측정 반복 | 300 |
| Seed | missing |
| Input shape | B1: [1,3,224,224], B4: [4,3,224,224], B16: [16,3,224,224] |
| Dtype | float32 |
| CPU 모델 / OS / Runtime 버전 | missing |

환경값은 JSON에 기록된 범위만 사용했다. CPU 모델, OS, PyTorch/ONNX Runtime 버전은 기록되지 않아 추정하지 않았다.

## 전체 측정 결과 표

Latency 단위 ms, Throughput 단위 samples/s다. Speedup은 같은 Batch의 `PyTorch mean / ONNX mean`이며 ONNX 행에 표시한다.

| Runtime | Batch | Mean | Median | P95 | Min | Max | Per-image | Throughput | ONNX speedup | Init | First |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ONNX Runtime CPU | 1 | 8.233 | 8.214 | 8.399 | 8.132 | 8.564 | 8.233 | 121.466 | 2.013 | 91.993 | 15.873 |
| PyTorch CPU | 1 | 16.574 | 16.439 | 17.205 | 16.208 | 20.352 | 16.574 | 60.336 | missing | 172.065 | 22.852 |
| ONNX Runtime CPU | 4 | 33.329 | 33.331 | 34.335 | 31.934 | 43.882 | 8.332 | 120.017 | 1.690 | 103.994 | 33.980 |
| PyTorch CPU | 4 | 56.319 | 54.636 | 65.428 | 50.948 | 84.045 | 14.080 | 71.024 | missing | 112.817 | 69.398 |
| ONNX Runtime CPU | 16 | 160.481 | 168.963 | 175.156 | 134.133 | 185.783 | 10.030 | 99.700 | 1.608 | 91.499 | 131.205 |
| PyTorch CPU | 16 | 258.133 | 257.750 | 272.119 | 219.905 | 284.733 | 16.133 | 61.984 | missing | 118.254 | 290.953 |

## PyTorch CPU와 ONNX Runtime CPU 비교

| Batch | PyTorch mean (ms) | ONNX mean (ms) | ONNX speedup | PyTorch samples/s | ONNX samples/s |
| --- | --- | --- | --- | --- | --- |
| 1 | 16.574 | 8.233 | 2.013 | 60.336 | 121.466 |
| 4 | 56.319 | 33.329 | 1.690 | 71.024 | 120.017 |
| 16 | 258.133 | 160.481 | 1.608 | 61.984 | 99.700 |

ONNX Runtime CPU가 측정된 모든 Batch에서 더 낮은 mean latency를 보였다. Speedup은 B1 2.013x, B4 1.690x, B16 1.608x다. Batch 증가와 함께 이 데이터의 ONNX 이점은 감소했다. 결과 파일만으로 커널 구현, 그래프 최적화, 메모리 접근 등 원인을 확정할 수 없다.

## Batch 1, 4, 16 확장성

Mean 배수, Per-image 변화, Throughput 변화는 각 Runtime의 B1 대비다.

| Runtime | Batch | Mean 배수 | Per-image 변화 | Throughput 변화 | ONNX speedup |
| --- | --- | --- | --- | --- | --- |
| PyTorch CPU | 1 | 1.000x | +0.0% | +0.0% | missing |
| ONNX Runtime CPU | 1 | 1.000x | +0.0% | +0.0% | 2.013 |
| PyTorch CPU | 4 | 3.398x | -15.0% | +17.7% | missing |
| ONNX Runtime CPU | 4 | 4.048x | +1.2% | -1.2% | 1.690 |
| PyTorch CPU | 16 | 15.575x | -2.7% | +2.7% | missing |
| ONNX Runtime CPU | 16 | 19.493x | +21.8% | -17.9% | 1.608 |

PyTorch는 B4에서 per-image latency가 B1 대비 -15.0%, throughput은 +17.7%였다. B16에서는 각각 -2.7%, +2.7%였다.

ONNX Runtime은 B4에서 per-image latency가 B1 대비 +1.2%, throughput은 -1.2%였다. B16에서는 각각 +21.8%, -17.9%였다.

## Latency와 Throughput 분석

PyTorch 최고 throughput은 B4의 71.024 samples/s다. ONNX Runtime 최고 throughput은 B1의 121.466 samples/s이며 B4는 비슷하지만 B16에서 99.700 samples/s로 낮아졌다. 큰 Batch가 자동으로 더 좋은 per-image latency나 throughput을 만들지 않았다.

Mean, median, P95, min, max는 JSON의 집계값을 그대로 사용했다. 원시 iteration latency가 없어 분산 형태를 복원할 수 없다.

## 초기화와 First Inference 분석

| Runtime | Batch | 초기화 종류 | 초기화 (ms) | Input load (ms) | NumPy→Tensor (ms) | First (ms) | First/Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ONNX Runtime CPU | 1 | session_loading | 91.993 | 0.495 | missing | 15.873 | 1.928x |
| PyTorch CPU | 1 | model_initialization | 172.065 | 0.289 | 0.022 | 22.852 | 1.379x |
| ONNX Runtime CPU | 4 | session_loading | 103.994 | 1.606 | missing | 33.980 | 1.020x |
| PyTorch CPU | 4 | model_initialization | 112.817 | 0.495 | 0.025 | 69.398 | 1.232x |
| ONNX Runtime CPU | 16 | session_loading | 91.499 | 2.290 | missing | 131.205 | 0.818x |
| PyTorch CPU | 16 | model_initialization | 118.254 | 2.567 | 0.030 | 290.953 | 1.127x |

PyTorch는 model initialization, ONNX Runtime은 session loading을 별도 항목으로 유지했다. ONNX JSON에 NumPy→Tensor 변환값은 없어 `missing`이다. First inference는 대체로 steady-state mean보다 높지만 ONNX B16은 낮다. 따라서 First inference를 항상 초기 실행 페널티로 해석할 수 없다.

## Memory와 RSS 분석

| Runtime | Batch | 생성 전 RSS | 생성 후 RSS | 생성 증가분 | Final RSS | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| ONNX Runtime CPU | 1 | 42.965 | 141.613 | 98.648 | 147.516 | 146.898 |
| PyTorch CPU | 1 | 638.367 | 699.691 | 61.324 | 722.352 | 730.781 |
| ONNX Runtime CPU | 4 | 44.668 | 143.316 | 98.648 | 166.824 | 166.594 |
| PyTorch CPU | 4 | 638.504 | 699.758 | 61.254 | 744.043 | 770.578 |
| ONNX Runtime CPU | 16 | 51.551 | 139.664 | 88.113 | 215.234 | 214.852 |
| PyTorch CPU | 16 | 638.504 | 699.789 | 61.285 | 766.051 | 863.910 |

생성 전/후 차이와 JSON의 model/session RSS delta는 Model 또는 Session 생성 증가분이다. Final/Peak RSS는 전체 프로세스 RSS다. PyTorch와 ONNX의 생성 전 RSS가 크게 다르므로 전체 RSS 차이를 Model 또는 Session 자체 메모리 차이로 간주하면 안 된다. 별도 프로세스의 런타임 기본 메모리와 측정 시점이 포함될 수 있다.

## Validation 분석

| Batch | Max difference | Mean difference | Allclose | Top-1 일치 |
| --- | --- | --- | --- | --- |
| 1 | 0.000004530 | 0.000000714 | 통과 | 통과 |
| 4 | 0.000004768 | 0.000000666 | 통과 | 통과 |
| 16 | 0.000004530 | 0.000000704 | 통과 | 통과 |

Batch별 Validation 전체 통과 여부: **전체 통과**. Allclose와 Top-1 일치를 각각 확인했다.

## 이상치와 가능한 원인

- B4 PyTorch max latency 84.045 ms는 median 54.636 ms보다 +53.8% 높다.
- B4 ONNX max latency 43.882 ms는 median 33.331 ms보다 +31.7% 높다.
- ONNX B16 mean 160.481 ms가 median 168.963 ms보다 낮고, first inference 131.205 ms도 mean보다 낮다.
- 모든 ONNX 결과에서 Final RSS가 기록된 Peak RSS보다 소폭 높다. Peak 측정 구간 또는 측정 시점 정의를 확인할 필요가 있다.
- 가능한 요인은 OS 스케줄링, CPU 주파수 변화, 캐시 상태, 백그라운드 부하, 메모리 할당 등이다. 원시 latency와 환경 로그가 없어 어느 원인도 확정할 수 없다.

## 실험 한계

- 원시 iteration latency가 없다. 표준편차, P99, Confidence Interval을 계산하지 않았다.
- CPU 모델, OS, Runtime/라이브러리 버전, 전원 정책, CPU affinity, 백그라운드 부하가 기록되지 않았다.
- Seed가 기록되지 않았다.
- 각 Runtime/Batch 조합이 집계 JSON 1개뿐이다. 실행 간 변동성과 재현성을 평가할 수 없다.
- PyTorch와 ONNX의 초기 프로세스 RSS가 달라 전체 RSS 절대값의 직접 비교가 제한된다.
- 결과는 ResNet-18, float32, thread 4, 입력 크기 224×224 범위에 한정된다.

## 다음 실험 제안

1. 원시 iteration latency와 실행별 타임스탬프를 저장하고 독립 실행을 여러 번 반복한다.
2. CPU 모델, OS, PyTorch/ONNX Runtime 버전, 전원 정책, affinity, 동시 부하를 기록한다.
3. Thread 수를 1/2/4/8로 바꾸고 Batch 1/4/16의 latency, throughput, speedup을 다시 측정한다.
4. RSS 샘플링 구간을 명시하고 초기 프로세스, 생성 직후, warm-up 후, 측정 중 peak를 같은 정의로 기록한다.
5. 입력을 여러 seed로 생성하고 각 Batch에서 Allclose와 Top-1 validation을 반복한다.
