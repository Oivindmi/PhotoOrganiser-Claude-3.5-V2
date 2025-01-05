import sys
import platform
import subprocess
import os
from typing import Dict, Optional, List, Tuple
import ctypes
from ctypes import wintypes


def get_windows_gpu_info() -> Optional[str]:
    try:
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                                capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error getting GPU info: {e}"


def get_all_env_paths() -> List[Tuple[str, str]]:
    relevant_vars = [
        'CUDA_PATH',
        'CUDA_PATH_V12_0',
        'PATH'
    ]

    return [(var, os.environ.get(var, 'Not found')) for var in relevant_vars]


def check_cuda_dlls() -> List[Tuple[str, bool]]:
    cuda_dlls = [
        'nvcuda.dll',
        'cudart64_120.dll',
        'cublas64_12.dll',
        'cufft64_12.dll'
    ]

    dll_status = []
    for dll in cuda_dlls:
        try:
            ctypes.WinDLL(dll)
            dll_status.append((dll, True))
        except WindowsError:
            dll_status.append((dll, False))
    return dll_status


def get_cuda_available_archs() -> List[str]:
    try:
        import torch
        return [str(arch) for arch in torch.cuda.get_arch_list()]
    except ImportError:
        return ["torch not installed"]
    except AttributeError:
        return ["torch installed but cuda not available"]


def check_opencv_cuda_modules():
    try:
        import cv2
        cuda_modules = []
        for attr in dir(cv2.cuda):
            if not attr.startswith('__'):
                cuda_modules.append(attr)
        return cuda_modules
    except AttributeError:
        return []


def main():
    print("\n=== Enhanced GPU Compatibility Check ===\n")

    print("System Information:")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print()

    print("GPU Information:")
    gpu_info = get_windows_gpu_info()
    print(gpu_info)
    print()

    print("NVIDIA Driver Information:")
    try:
        nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(nvidia_smi.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found. NVIDIA driver might not be installed.")
    print()

    print("CUDA Environment Variables:")
    for var, path in get_all_env_paths():
        print(f"{var}: {path}")
    print()

    print("CUDA DLL Status:")
    for dll, status in check_cuda_dlls():
        print(f"{dll}: {'Found' if status else 'Not Found'}")
    print()

    print("OpenCV Information:")
    try:
        import cv2
        print(f"OpenCV Version: {cv2.__version__}")
        cuda_enabled = cv2.cuda.getCudaEnabledDeviceCount() > 0
        print(f"CUDA Enabled: {cuda_enabled}")
        if cuda_enabled:
            print(f"Number of CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
            device = cv2.cuda.getDevice()
            print(f"Current device: {device}")
    except ImportError:
        print("OpenCV not installed")
    except AttributeError:
        print("OpenCV installed but CUDA support not available")
    print()

    print("Available CUDA Modules in OpenCV:")
    cuda_modules = check_opencv_cuda_modules()
    if cuda_modules:
        print("\n".join(cuda_modules))
    else:
        print("No CUDA modules found in OpenCV")
    print()

    try:
        import torch
        print("\nPyTorch CUDA Information:")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Available architectures: {get_cuda_available_archs()}")
    except ImportError:
        print("\nPyTorch not installed")


if __name__ == "__main__":
    main()