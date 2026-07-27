import statistics
import numpy as np
import time

def benchmark(inference_fn,warming_count=10,measured_count=300):
    for _ in range(warming_count):
        inference_fn()

    # 실제 측정
    times = []

    for _ in range(measured_count):
        start_time = time.perf_counter()
        inference_fn()
        end_time = time.perf_counter()

        times.append((end_time - start_time) * 1000)

    return times


def print_statistics(times):
    print(f"Mean: {statistics.mean(times):.4f} ms")
    print(f"Median: {statistics.median(times):.4f} ms")
    print(f"P95: {np.percentile(times, 95):.4f} ms")
    print(f"Min: {min(times):.4f} ms")
    print(f"Max: {max(times):.4f} ms")
    