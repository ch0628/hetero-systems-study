# CPU Benchmark Report

## 데이터 범위

- 분석 Batch: 1, 4, 16
- 분석 Intra-op thread: 1, 2, 4
- 성능 결과: 18개
- Validation 결과: 9개
- ONNX Export 결과: 3개

## 재현 환경

| 항목 | 기록값 |
| --- | --- |
| Hostname | diaho |
| OS | Linux-6.8.0-124-generic-x86_64-with-glibc2.35 |
| Kernel | 6.8.0-124-generic |
| Architecture | x86_64 |
| CPU | x86_64 |
| Physical / Logical cores | 4 / 8 |
| RAM (GB) | 15.348 |
| Python | 3.10.12 |
| PyTorch | 2.13.0 |
| Torchvision | 0.28.0 |
| ONNX | 1.22.0 |
| ONNX Runtime | 1.23.2 |
| Virtual environment | /home/steelho/study/hetero-systems-study/.venv |
| Git commit | 16765a13b633cf278ae95efc7bcaf23079a7a128 |

## 전체 측정 결과

Latency 단위는 ms, Throughput 단위는 images/s다. ONNX speedup은 동일 Batch·Thread 조건에서 PyTorch mean / ONNX mean이다.

| Runtime | Threads | Batch | Mean | Median | P95 | P99 | Std | 95% CI | Per-image | Throughput | ONNX speedup | Raw count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ONNX Runtime CPU | 1 | 1 | 29.085 | 28.954 | 29.495 | 31.167 | 0.389 | [29.041, 29.129] | 29.085 | 34.382 | 1.465 | 300 |
| PyTorch CPU | 1 | 1 | 42.617 | 42.408 | 43.753 | 47.438 | 0.856 | [42.520, 42.714] | 42.617 | 23.465 | missing | 300 |
| ONNX Runtime CPU | 1 | 4 | 114.304 | 114.004 | 115.085 | 119.595 | 1.499 | [114.134, 114.473] | 28.576 | 34.994 | 1.366 | 300 |
| PyTorch CPU | 1 | 4 | 156.180 | 155.598 | 159.206 | 168.026 | 2.560 | [155.891, 156.470] | 39.045 | 25.611 | missing | 300 |
| ONNX Runtime CPU | 1 | 16 | 468.967 | 471.752 | 476.205 | 481.002 | 6.747 | [468.204, 469.731] | 29.310 | 34.118 | 1.374 | 300 |
| PyTorch CPU | 1 | 16 | 644.195 | 643.512 | 657.650 | 667.794 | 8.770 | [643.202, 645.187] | 40.262 | 24.837 | missing | 300 |
| ONNX Runtime CPU | 2 | 1 | 17.071 | 17.026 | 17.161 | 17.413 | 0.759 | [16.985, 17.157] | 17.071 | 58.578 | 1.446 | 300 |
| PyTorch CPU | 2 | 1 | 24.691 | 24.612 | 25.096 | 27.262 | 0.473 | [24.638, 24.745] | 24.691 | 40.500 | missing | 300 |
| ONNX Runtime CPU | 2 | 4 | 65.275 | 65.746 | 66.760 | 70.856 | 1.513 | [65.104, 65.446] | 16.319 | 61.279 | 1.411 | 300 |
| PyTorch CPU | 2 | 4 | 92.093 | 90.245 | 95.536 | 103.789 | 3.509 | [91.696, 92.490] | 23.023 | 43.435 | missing | 300 |
| ONNX Runtime CPU | 2 | 16 | 263.053 | 261.999 | 281.220 | 282.838 | 7.542 | [262.199, 263.906] | 16.441 | 60.824 | 1.452 | 300 |
| PyTorch CPU | 2 | 16 | 382.025 | 378.895 | 400.893 | 409.385 | 11.022 | [380.778, 383.272] | 23.877 | 41.882 | missing | 300 |
| ONNX Runtime CPU | 4 | 1 | 10.665 | 10.924 | 11.004 | 11.273 | 0.656 | [10.591, 10.739] | 10.665 | 93.767 | 1.784 | 300 |
| PyTorch CPU | 4 | 1 | 19.029 | 19.227 | 20.305 | 22.414 | 1.398 | [18.870, 19.187] | 19.029 | 52.553 | missing | 300 |
| ONNX Runtime CPU | 4 | 4 | 43.054 | 42.882 | 44.459 | 45.850 | 0.938 | [42.948, 43.161] | 10.764 | 92.906 | 1.498 | 300 |
| PyTorch CPU | 4 | 4 | 64.474 | 64.432 | 66.951 | 76.414 | 3.175 | [64.115, 64.834] | 16.119 | 62.040 | missing | 300 |
| ONNX Runtime CPU | 4 | 16 | 172.169 | 171.008 | 177.932 | 183.848 | 2.921 | [171.838, 172.499] | 10.761 | 92.932 | 1.484 | 300 |
| PyTorch CPU | 4 | 16 | 255.555 | 254.073 | 264.820 | 279.982 | 5.542 | [254.928, 256.182] | 15.972 | 62.609 | missing | 300 |

## Runtime 비교

| Threads | Batch | PyTorch mean | ONNX mean | ONNX speedup | PyTorch images/s | ONNX images/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 42.617 | 29.085 | 1.465 | 23.465 | 34.382 |
| 1 | 4 | 156.180 | 114.304 | 1.366 | 25.611 | 34.994 |
| 1 | 16 | 644.195 | 468.967 | 1.374 | 24.837 | 34.118 |
| 2 | 1 | 24.691 | 17.071 | 1.446 | 40.500 | 58.578 |
| 2 | 4 | 92.093 | 65.275 | 1.411 | 43.435 | 61.279 |
| 2 | 16 | 382.025 | 263.053 | 1.452 | 41.882 | 60.824 |
| 4 | 1 | 19.029 | 10.665 | 1.784 | 52.553 | 93.767 |
| 4 | 4 | 64.474 | 43.054 | 1.498 | 62.040 | 92.906 |
| 4 | 16 | 255.555 | 172.169 | 1.484 | 62.609 | 92.932 |

## Thread 확장성

Thread latency speedup은 각 Runtime·Batch에서 가장 작은 Thread 수를 기준으로 계산한다.

| Runtime | Batch | Threads | Mean | Latency speedup | Throughput 변화 | P99 |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch CPU | 1 | 1 | 42.617 | 1.000 | +0.0% | 47.438 |
| PyTorch CPU | 1 | 2 | 24.691 | 1.726 | +72.6% | 27.262 |
| PyTorch CPU | 1 | 4 | 19.029 | 2.240 | +124.0% | 22.414 |
| PyTorch CPU | 4 | 1 | 156.180 | 1.000 | +0.0% | 168.026 |
| PyTorch CPU | 4 | 2 | 92.093 | 1.696 | +69.6% | 103.789 |
| PyTorch CPU | 4 | 4 | 64.474 | 2.422 | +142.2% | 76.414 |
| PyTorch CPU | 16 | 1 | 644.195 | 1.000 | +0.0% | 667.794 |
| PyTorch CPU | 16 | 2 | 382.025 | 1.686 | +68.6% | 409.385 |
| PyTorch CPU | 16 | 4 | 255.555 | 2.521 | +152.1% | 279.982 |
| ONNX Runtime CPU | 1 | 1 | 29.085 | 1.000 | +0.0% | 31.167 |
| ONNX Runtime CPU | 1 | 2 | 17.071 | 1.704 | +70.4% | 17.413 |
| ONNX Runtime CPU | 1 | 4 | 10.665 | 2.727 | +172.7% | 11.273 |
| ONNX Runtime CPU | 4 | 1 | 114.304 | 1.000 | +0.0% | 119.595 |
| ONNX Runtime CPU | 4 | 2 | 65.275 | 1.751 | +75.1% | 70.856 |
| ONNX Runtime CPU | 4 | 4 | 43.054 | 2.655 | +165.5% | 45.850 |
| ONNX Runtime CPU | 16 | 1 | 468.967 | 1.000 | +0.0% | 481.002 |
| ONNX Runtime CPU | 16 | 2 | 263.053 | 1.783 | +78.3% | 282.838 |
| ONNX Runtime CPU | 16 | 4 | 172.169 | 2.724 | +172.4% | 183.848 |

## Batch 확장성

Per-image와 Throughput 변화는 동일 Runtime·Thread에서 가장 작은 Batch를 기준으로 계산한다.

| Runtime | Threads | Batch | Mean | Per-image 변화 | Throughput 변화 |
| --- | --- | --- | --- | --- | --- |
| PyTorch CPU | 1 | 1 | 42.617 | +0.0% | +0.0% |
| PyTorch CPU | 1 | 4 | 156.180 | -8.4% | +9.1% |
| PyTorch CPU | 1 | 16 | 644.195 | -5.5% | +5.8% |
| PyTorch CPU | 2 | 1 | 24.691 | +0.0% | +0.0% |
| PyTorch CPU | 2 | 4 | 92.093 | -6.8% | +7.2% |
| PyTorch CPU | 2 | 16 | 382.025 | -3.3% | +3.4% |
| PyTorch CPU | 4 | 1 | 19.029 | +0.0% | +0.0% |
| PyTorch CPU | 4 | 4 | 64.474 | -15.3% | +18.1% |
| PyTorch CPU | 4 | 16 | 255.555 | -16.1% | +19.1% |
| ONNX Runtime CPU | 1 | 1 | 29.085 | +0.0% | +0.0% |
| ONNX Runtime CPU | 1 | 4 | 114.304 | -1.8% | +1.8% |
| ONNX Runtime CPU | 1 | 16 | 468.967 | +0.8% | -0.8% |
| ONNX Runtime CPU | 2 | 1 | 17.071 | +0.0% | +0.0% |
| ONNX Runtime CPU | 2 | 4 | 65.275 | -4.4% | +4.6% |
| ONNX Runtime CPU | 2 | 16 | 263.053 | -3.7% | +3.8% |
| ONNX Runtime CPU | 4 | 1 | 10.665 | +0.0% | +0.0% |
| ONNX Runtime CPU | 4 | 4 | 43.054 | +0.9% | -0.9% |
| ONNX Runtime CPU | 4 | 16 | 172.169 | +0.9% | -0.9% |

## ONNX Export 기록

| Batch | Opset | Export ms | Size MB | Checker | Nodes | Operator counts |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 17 | 2215.761 | 0.090 | 통과 | 49 | {"Add":8,"Conv":20,"Gemm":1,"MaxPool":1,"ReduceMean":1,"Relu":17,"Reshape":1} |
| 16 | 17 | 2065.214 | 0.090 | 통과 | 49 | {"Add":8,"Conv":20,"Gemm":1,"MaxPool":1,"ReduceMean":1,"Relu":17,"Reshape":1} |
| 4 | 17 | 2054.870 | 0.090 | 통과 | 49 | {"Add":8,"Conv":20,"Gemm":1,"MaxPool":1,"ReduceMean":1,"Relu":17,"Reshape":1} |

## 출력 검증

| Batch | Threads | Max diff | Mean diff | Allclose | Top-1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.000004530 | 0.000000714 | 통과 | 통과 |
| 4 | 1 | 0.000004768 | 0.000000666 | 통과 | 통과 |
| 16 | 1 | 0.000004530 | 0.000000704 | 통과 | 통과 |
| 1 | 2 | 0.000004530 | 0.000000714 | 통과 | 통과 |
| 4 | 2 | 0.000004768 | 0.000000666 | 통과 | 통과 |
| 16 | 2 | 0.000004530 | 0.000000704 | 통과 | 통과 |
| 1 | 4 | 0.000004530 | 0.000000714 | 통과 | 통과 |
| 4 | 4 | 0.000004768 | 0.000000666 | 통과 | 통과 |
| 16 | 4 | 0.000004530 | 0.000000704 | 통과 | 통과 |

전체 Batch·Thread 검증 결과: **전체 통과**

## 조건별 최저 Mean Runtime

- T1, B1: ONNX Runtime CPU mean 29.085 ms
- T1, B4: ONNX Runtime CPU mean 114.304 ms
- T1, B16: ONNX Runtime CPU mean 468.967 ms
- T2, B1: ONNX Runtime CPU mean 17.071 ms
- T2, B4: ONNX Runtime CPU mean 65.275 ms
- T2, B16: ONNX Runtime CPU mean 263.053 ms
- T4, B1: ONNX Runtime CPU mean 10.665 ms
- T4, B4: ONNX Runtime CPU mean 43.054 ms
- T4, B16: ONNX Runtime CPU mean 172.169 ms

## 데이터 품질 점검

- 원시 Latency 개수 불일치 또는 누락: 0개
- P99·표준편차·95% CI 누락: 0개
- 기대 Validation 조합: 9개, 실제: 9개

## 해석 시 주의사항

- Thread 수가 많다고 항상 Mean, P99, Throughput이 개선되는 것은 아니다.
- Runtime 우위는 Batch와 Thread 조건을 고정한 뒤 비교해야 한다.
- 95% CI는 한 실행 안의 iteration 표본에 대한 구간이며, 독립 프로세스 반복 간 재현성을 대신하지 않는다.
- RSS는 전체 프로세스와 Model/Session 생성 증가분을 구분해서 해석해야 한다.
- ONNX Checker 통과는 실행 가능성 검증이지 성능 향상 보장이 아니다.

## 다음 단계

1. 같은 Batch·Thread 조합을 독립 프로세스로 여러 번 반복해 실행 간 분산을 측정한다.
2. CPU affinity, 전원 정책, 백그라운드 부하를 기록한다.
3. Thread별 CPU utilization을 함께 수집한다.
4. Fixed Shape와 Dynamic Shape 비교로 확장한다.
