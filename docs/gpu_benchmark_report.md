# GPU 벤치마크 분석 보고서

## 데이터 범위

- 성능 로그: `results/gpu/*.txt` (6개)
- 검증 결과: `benchmark/results/validation/*.json` (6개)
- 원본 파일은 읽기만 했으며 수정하지 않았다.
- 값이 기록되지 않은 항목은 `missing`으로 표기했다.

## 실험 환경

| 항목 | 기록값 |
| --- | --- |
| 호스트 | moana-r3 |
| GPU | NVIDIA GeForce RTX 3090 |
| 요청 장치/Provider | cuda; cuda |
| ONNX 실제 Provider | ['CUDAExecutionProvider', 'CPUExecutionProvider'] |
| PyTorch | 2.4.0+cu121 |
| ONNX Runtime 버전 | missing |
| CUDA runtime | 12.1 |
| cuDNN | 90100 |
| 입력 shape | (1, 3, 224, 224); (16, 3, 224, 224); (4, 3, 224, 224) |
| 출력 shape | (1, 1000); (16, 1000); (4, 1000) |
| 모델명 | missing |
| 정밀도 | missing |
| warm-up/측정 반복 수 | missing |

환경 항목은 로그에 명시된 값만 사용했다. `Actual providers`에 CPU fallback Provider도 함께 등록되어 있지만, 연산별 Provider 배치는 로그에 없어 확인할 수 없다.

## 전체 측정 결과

### GPU-only latency

| Runtime | Batch | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) | Throughput (samples/s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch CUDA | 1 | 4.1009 | 4.1063 | 4.1698 | 4.0436 | 4.2341 | 243.85 |
| ONNX Runtime CUDA | 1 | 0.9938 | 0.9883 | 1.0121 | 0.9695 | 1.1683 | 1006.24 |
| PyTorch CUDA | 4 | 4.2221 | 4.2100 | 4.2966 | 4.1793 | 4.3457 | 947.40 |
| ONNX Runtime CUDA | 4 | 1.3773 | 1.3753 | 1.3792 | 1.3642 | 1.5090 | 2904.23 |
| PyTorch CUDA | 16 | 4.2338 | 4.2267 | 4.3125 | 4.1759 | 5.9654 | 3779.11 |
| ONNX Runtime CUDA | 16 | 3.8386 | 3.8335 | 3.8705 | 3.7683 | 3.9903 | 4168.19 |

### End-to-End latency

| Runtime | Batch | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) | Throughput (samples/s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch CUDA | 1 | 4.3067 | 4.2926 | 4.3923 | 4.2531 | 4.4503 | 232.20 |
| ONNX Runtime CUDA | 1 | 1.1037 | 1.0931 | 1.2201 | 1.0768 | 1.3846 | 906.04 |
| PyTorch CUDA | 4 | 4.5772 | 4.5509 | 4.6487 | 4.5079 | 9.2311 | 873.90 |
| ONNX Runtime CUDA | 4 | 1.6853 | 1.7273 | 1.7663 | 1.5732 | 1.8871 | 2373.46 |
| PyTorch CUDA | 16 | 5.1350 | 5.1345 | 5.2176 | 5.0254 | 5.3717 | 3115.87 |
| ONNX Runtime CUDA | 16 | 4.5941 | 4.5680 | 4.8621 | 4.4181 | 4.9899 | 3482.73 |

### 초기화와 first inference

| Runtime | Batch | 초기화 종류 | 초기화 (ms) | First inference (ms) |
| --- | --- | --- | --- | --- |
| PyTorch CUDA | 1 | Model initialization | 764.9986 | 2317.7179 |
| ONNX Runtime CUDA | 1 | Session loading | 1069.2381 | 962.8064 |
| PyTorch CUDA | 4 | Model initialization | 562.5119 | 549.8177 |
| ONNX Runtime CUDA | 4 | Session loading | 308.0260 | 2481.1656 |
| PyTorch CUDA | 16 | Model initialization | 400.4250 | 551.8982 |
| ONNX Runtime CUDA | 16 | Session loading | 308.6302 | 2579.1292 |

Throughput은 각 구간의 mean latency에 대해 `batch / (mean_latency_ms / 1000)`으로 계산했다. 초기화와 first inference는 steady-state latency/throughput 계산에서 제외했다.

## PyTorch와 ONNX Runtime 비교

| Batch | PyTorch GPU mean (ms) | ONNX GPU mean (ms) | ONNX GPU speedup | PyTorch E2E mean (ms) | ONNX E2E mean (ms) | ONNX E2E speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4.1009 | 0.9938 | 4.126 | 4.3067 | 1.1037 | 3.902 |
| 4 | 4.2221 | 1.3773 | 3.065 | 4.5772 | 1.6853 | 2.716 |
| 16 | 4.2338 | 3.8386 | 1.103 | 5.1350 | 4.5941 | 1.118 |

- ONNX Runtime이 모든 batch에서 더 낮은 mean latency를 기록했다. GPU-only speedup 범위: 1.103x–4.126x.
- End-to-End speedup 범위: 1.118x–3.902x.
- batch가 커질수록 ONNX Runtime의 우위가 감소했다. 이는 측정된 경향이며 원인은 이 로그만으로 확정할 수 없다.

## Batch 1, 4, 16 확장성

| Runtime | Batch | GPU mean (ms) | GPU latency vs B1 | GPU throughput | GPU throughput vs B1 | E2E mean (ms) | E2E throughput | E2E throughput vs B1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch CUDA | 1 | 4.1009 | 0.00% | 243.85 | 1.00x | 4.3067 | 232.20 | 1.00x |
| PyTorch CUDA | 4 | 4.2221 | 2.96% | 947.40 | 3.89x | 4.5772 | 873.90 | 3.76x |
| PyTorch CUDA | 16 | 4.2338 | 3.24% | 3779.11 | 15.50x | 5.1350 | 3115.87 | 13.42x |
| ONNX Runtime CUDA | 1 | 0.9938 | 0.00% | 1006.24 | 1.00x | 1.1037 | 906.04 | 1.00x |
| ONNX Runtime CUDA | 4 | 1.3773 | 38.59% | 2904.23 | 2.89x | 1.6853 | 2373.46 | 2.62x |
| ONNX Runtime CUDA | 16 | 3.8386 | 286.25% | 4168.19 | 4.14x | 4.5941 | 3482.73 | 3.84x |

## Latency와 throughput 분석

- PyTorch CUDA: batch 1→16에서 GPU-only mean latency는 4.1009 ms→4.2338 ms, throughput은 243.85→3779.11 samples/s였다.
- ONNX Runtime CUDA: batch 1→16에서 GPU-only mean latency는 0.9938 ms→3.8386 ms, throughput은 1006.24→4168.19 samples/s였다.
- PyTorch CUDA, batch 1: E2E와 GPU-only mean 차이는 0.2058 ms (5.02%).
- ONNX Runtime CUDA, batch 1: E2E와 GPU-only mean 차이는 0.1099 ms (11.06%).
- PyTorch CUDA, batch 4: E2E와 GPU-only mean 차이는 0.3551 ms (8.41%).
- ONNX Runtime CUDA, batch 4: E2E와 GPU-only mean 차이는 0.3080 ms (22.36%).
- PyTorch CUDA, batch 16: E2E와 GPU-only mean 차이는 0.9012 ms (21.29%).
- ONNX Runtime CUDA, batch 16: E2E와 GPU-only mean 차이는 0.7555 ms (19.68%).

GPU-only는 장치 실행 중심, End-to-End는 호출·전송·출력 처리 오버헤드를 포함한 지표다. 따라서 두 throughput은 서로 다른 운영 경계를 나타낸다.

## Validation 분석

| Profile | Batch | Max difference | Mean difference | Allclose | Top-1 일치 | rtol | atol | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cuda | 1 | 0.00532269477844238 | 0.000812865153420717 | false | true | 0.0001 | 1e-05 | benchmark/results/validation/validation_cuda_b1.json |
| cuda | 4 | 0.00621175765991211 | 0.00113086972851306 | false | true | 0.0001 | 1e-05 | benchmark/results/validation/validation_cuda_b4.json |
| cuda | 16 | 0.00786256790161133 | 0.0012278911890462 | false | true | 0.0001 | 1e-05 | benchmark/results/validation/validation_cuda_b16.json |
| unspecified_device | 1 | 4.52995300292969e-06 | 7.13640815774852e-07 | true | true | 0.0001 | 1e-05 | benchmark/results/validation/validation_b1.json |
| unspecified_device | 4 | 4.76837158203125e-06 | 6.6589899461178e-07 | true | true | 0.0001 | 1e-05 | benchmark/results/validation/validation_b4.json |
| unspecified_device | 16 | 4.52995300292969e-06 | 7.04321621469717e-07 | true | true | 0.0001 | 1e-05 | benchmark/results/validation/validation_b16.json |

- CUDA profile: allclose 통과 0/3, Top-1 일치 3/3.
- 장치 미기록 profile: allclose 통과 3/3, Top-1 일치 3/3.
- CUDA profile은 모든 batch에서 설정된 `rtol=0.0001`, `atol=1e-05` 기준 allclose에 실패했지만 Top-1 class는 모두 일치했다.
- 장치 미기록 profile은 모든 batch에서 allclose와 Top-1 일치가 모두 true였다. 해당 JSON에 장치 정보가 없어 CPU 결과라고 단정하지 않았다.

## 이상치와 가능한 원인

- PyTorch CUDA, batch 16, GPU-only: Max 5.9654 ms가 P95 4.3125 ms의 1.38배. 이상치 후보.
- PyTorch CUDA, batch 4, End-to-End: Max 9.2311 ms가 P95 4.6487 ms의 1.99배. 이상치 후보.
- 가능한 원인: CUDA 비동기 동기화 지점, 최초 kernel/context 준비, 메모리 할당·복사, OS scheduling, 다른 프로세스의 GPU 점유, 온도·클럭 변동. 로그만으로 어느 원인인지 확정할 수 없다.
- first inference와 initialization이 batch/실행별로 크게 달랐다. 실행 순서, 프로세스 재사용, cache 상태가 기록되지 않아 직접 비교에 주의가 필요하다.
- CUDA validation 차이의 가능한 원인에는 backend별 연산 순서, kernel 구현, TF32/FP32 처리, 누적 반올림이 있다. 정밀도 설정과 연산별 오차 자료가 없어 확정할 수 없다.

## 실험 한계

- 모델명, ONNX Runtime 버전, 정밀도 설정, warm-up 횟수, 측정 반복 수가 로그에 없다.
- 원시 iteration latency가 없어 분포 모양, 표준편차, confidence interval을 재계산할 수 없다.
- 단일 호스트와 단일 GPU 결과라 다른 GPU·드라이버·소프트웨어 조합으로 일반화할 수 없다.
- 실행 순서, 독립 프로세스 여부, cache 초기화, GPU clock·temperature·utilization이 기록되지 않았다.
- `Actual providers`는 등록 Provider만 보여 주며 각 ONNX node가 CUDA에서 실행됐는지 증명하지 않는다.
- validation 두 profile 중 장치 미기록 파일은 실행 장치를 확정할 수 없다.

## 다음 실험 제안

1. 모델명, ONNX Runtime/driver 버전, precision, warm-up·반복 수, seed를 로그에 추가한다.
2. runtime·batch 조합을 독립 프로세스에서 여러 번 무작위 순서로 실행하고 원시 iteration latency를 저장한다.
3. mean/median/P95뿐 아니라 표준편차와 95% confidence interval을 계산한다.
4. GPU utilization, clock, temperature, power, memory 사용량을 측정 구간과 함께 기록한다.
5. ONNX Runtime profiling으로 node별 Execution Provider 배치를 확인한다.
6. TF32 허용 여부와 FP32/FP16 설정을 고정해 CUDA allclose 실패 원인을 분리한다.
7. batch 2, 8, 32 및 saturation/OOM 지점까지 확장해 throughput 포화 구간을 찾는다.
8. first inference를 CUDA context 준비, model/session 초기화, memory allocation, kernel 준비 단계로 분리 측정한다.
