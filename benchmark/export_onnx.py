from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from torchvision.models import ResNet18_Weights, resnet18


def external_data_paths(model: onnx.ModelProto, model_path: Path) -> list[Path]:
    paths: list[Path] = []
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for item in initializer.external_data:
            if item.key == "location" and item.value:
                paths.append(model_path.parent / item.value)
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed-shape ResNet18 ONNX models")
    parser.add_argument("--batch", type=int, required=True, choices=[1, 4, 16])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    benchmark_dir = Path(__file__).resolve().parent
    input_dir = benchmark_dir / "data" / "inputs"
    model_dir = benchmark_dir / "data" / "onnx_models"
    metadata_dir = benchmark_dir / "results" / "export"
    input_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / f"input_b{args.batch}.npy"
    model_path = model_dir / f"resnet18_b{args.batch}.onnx"
    metadata_path = metadata_dir / f"onnx_export_b{args.batch}.json"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()
    input_tensor = torch.randn(args.batch, 3, 224, 224, dtype=torch.float32)
    np.save(input_path, input_tensor.numpy())

    start = time.perf_counter()
    torch.onnx.export(
        model,
        input_tensor,
        model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    export_time_ms = (time.perf_counter() - start) * 1000.0

    onnx_model = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(onnx_model)

    op_counts: dict[str, int] = {}
    for node in onnx_model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    external_paths = external_data_paths(onnx_model, model_path)
    artifact_paths = [model_path, *external_paths]
    total_size_bytes = sum(path.stat().st_size for path in artifact_paths if path.exists())

    metadata = {
        "model": "resnet18",
        "weights": "ResNet18_Weights.DEFAULT",
        "seed": args.seed,
        "batch_size": args.batch,
        "input_shape": list(input_tensor.shape),
        "input_dtype": str(input_tensor.dtype),
        "input_path": str(input_path),
        "model_path": str(model_path),
        "opset": args.opset,
        "export_time_ms": export_time_ms,
        "onnx_checker_passed": True,
        "graph_node_count": len(onnx_model.graph.node),
        "graph_initializer_count": len(onnx_model.graph.initializer),
        "op_counts": dict(sorted(op_counts.items())),
        "main_onnx_file_size_bytes": model_path.stat().st_size,
        "external_data_files": [str(path) for path in external_paths],
        "total_model_artifact_size_bytes": total_size_bytes,
        "total_model_artifact_size_mb": total_size_bytes / (1024**2),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Batch size:", args.batch)
    print("Input shape:", tuple(input_tensor.shape))
    print("Opset:", args.opset)
    print(f"Export time: {export_time_ms:.4f} ms")
    print("ONNX checker: PASS")
    print("Graph nodes:", len(onnx_model.graph.node))
    print(f"Total model artifact size: {metadata['total_model_artifact_size_mb']:.2f} MB")
    print("Input saved:", input_path)
    print("ONNX model saved:", model_path)
    print("Export metadata saved:", metadata_path)


if __name__ == "__main__":
    main()
