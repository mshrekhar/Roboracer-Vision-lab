import torch
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

# --- 1. CONFIGURATION ---
MODEL_PTH = "model.pth"        # Your trained weights
ONNX_PATH = "model.onnx"       # Intermediate file
ENGINE_PATH_FP32 = "model_fp32.engine"
ENGINE_PATH_FP16 = "model_fp16.engine"
INPUT_SHAPE = (1, 3, 224, 224) # Adjust to your YOLO input size

def export_onnx():
    print(">>> Stage 1: Exporting PyTorch to ONNX...")
    # Load your model architecture (ensure your Model class is defined/imported)
    # from your_model_file import YourModelClass
    # model = YourModelClass(channels=...) 
    
    # For this example, assuming a standard loading procedure:
    model = torch.load(MODEL_PTH) 
    model.eval().cuda()
    
    dummy_input = torch.randn(INPUT_SHAPE).cuda()
    
    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_PATH, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output']
    )
    print(f"Done. ONNX saved to {ONNX_PATH}")

def build_engine(use_fp16=False):
    precision = "FP16" if use_fp16 else "FP32"
    print(f">>> Stage 2: Building {precision} TensorRT Engine...")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # Required for TRT 8.5 (Explicit Batch Flag)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Set workspace memory (e.g., 1GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if use_fp16:
        if not builder.platform_has_fast_fp16:
            print("Warning: This Jetson does not support fast FP16.")
        config.set_flag(trt.Flag.FP16)

    # Read ONNX
    with open(ONNX_PATH, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Build and Serialize
    engine_bytes = builder.build_serialized_network(network, config)
    
    save_path = ENGINE_PATH_FP16 if use_fp16 else ENGINE_PATH_FP32
    with open(save_path, "wb") as f:
        f.write(engine_bytes)
    
    print(f"Successfully saved engine to {save_path}")

if __name__ == "__main__":
    # Part A: PyTorch to ONNX (Requires PyTorch)
    if not os.path.exists(ONNX_PATH):
        export_onnx()
    
    # Part B: ONNX to TensorRT (Only uses TRT library)
    build_engine(use_fp16=False) # Build FP32
    build_engine(use_fp16=True)  # Build FP16