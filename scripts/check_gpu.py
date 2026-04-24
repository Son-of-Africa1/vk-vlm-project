import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("Total VRAM (GB):", round(props.total_memory / 1024**3, 2))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
else:
    print("PyTorch is not seeing your NVIDIA GPU.")