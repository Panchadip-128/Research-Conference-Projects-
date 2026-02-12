import torch

def format_pytorch_version(version):
    """
    Formats the PyTorch version string (e.g., '2.0.1+cu118' -> '2.0.1').
    """
    return version.split('+')[0]

def get_device():
    """
    Returns the available device (CUDA or CPU).
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_system_info():
    """
    Prints detected PyTorch and CUDA versions.
    """
    TORCH_version = torch.__version__
    CUDA_version = torch.version.cuda
    print(f"Detected PyTorch: {TORCH_version}")
    print(f"Detected CUDA: {CUDA_version}")
