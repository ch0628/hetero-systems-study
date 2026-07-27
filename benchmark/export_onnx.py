import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights


parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch",
    type=int,
    required=True,
    choices=[1, 4, 16],
    help="Batch size used for ONNX export",
)

args = parser.parse_args()


BENCHMARK_DIR = Path(__file__).resolve().parent

INPUT_DIR = BENCHMARK_DIR / "data" / "inputs"
MODEL_DIR = BENCHMARK_DIR / "data" / "onnx_models"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = INPUT_DIR / f"input_b{args.batch}.npy"
MODEL_PATH = MODEL_DIR / f"resnet18_b{args.batch}.onnx"


torch.manual_seed(42)

model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)
model.eval()


input_tensor = torch.randn(
    args.batch,
    3,
    224,
    224,
)


np.save(
    INPUT_PATH,
    input_tensor.numpy(),
)


torch.onnx.export(
    model,
    input_tensor,
    MODEL_PATH,
    input_names=["input"],
    output_names=["output"],
)


print("Batch size:", args.batch)
print("Input shape:", tuple(input_tensor.shape))
print("Input saved:", INPUT_PATH)
print("ONNX model saved:", MODEL_PATH)