from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "input.npy"
PYTORCH_OUTPUT_PATH = ROOT_DIR / "pytorch_output.npy"
MODEL_PATH = ROOT_DIR / "resnet18.onnx"

input_array = np.load(INPUT_PATH)
pytorch_output = np.load(PYTORCH_OUTPUT_PATH)


options = ort.SessionOptions()      
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)


input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print("ONNX input name:", input_info.name)
print("ONNX input shape:", input_info.shape)
print("ONNX input type:", input_info.type)

print("ONNX output name:", output_info.name)
print("ONNX output shape:", output_info.shape)
print("ONNX output type:", output_info.type)


onnx_output = session.run(
    None,
    {input_info.name: input_array},
)[0]


max_absolute_difference = np.max(
    np.abs(pytorch_output - onnx_output)
)

all_close = np.allclose(
    pytorch_output,
    onnx_output,
    rtol=1e-4,
    atol=1e-5,
)

pytorch_top1 = np.argmax(pytorch_output, axis=1)
onnx_top1 = np.argmax(onnx_output, axis=1)


print("PyTorch output shape:", pytorch_output.shape)
print("ONNX output shape:", onnx_output.shape)
print("Max absolute difference:", max_absolute_difference)
print("Output all close:", all_close)
print("PyTorch Top-1:", pytorch_top1)
print("ONNX Top-1:", onnx_top1)
print("Same Top-1:", np.array_equal(pytorch_top1, onnx_top1))