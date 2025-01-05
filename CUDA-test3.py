import subprocess
import os
from typing import Dict, Optional


def get_nvidia_smi_info() -> Optional[Dict[str, str]]:
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return {"output": result.stdout}
        return None
    except FileNotFoundError:
        return None


def get_nvcc_version() -> Optional[str]:
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        return None
    except FileNotFoundError:
        return None


def test_cuda() -> None:
    print("\nTesting CUDA Installation\n" + "=" * 50)

    # Check CUDA environment variables
    cuda_path = os.environ.get('CUDA_PATH')
    cuda_home = os.environ.get('CUDA_HOME')

    print("\nCUDA Environment Variables:")
    print(f"  CUDA_PATH: {cuda_path or 'Not set'}")
    print(f"  CUDA_HOME: {cuda_home or 'Not set'}")

    # Check NVCC
    print("\nNVCC Version:")
    nvcc_version = get_nvcc_version()
    if nvcc_version:
        print(f"  {nvcc_version.split('release')[1].split('V')[0].strip()}")
    else:
        print("  NVCC not found in PATH")

    # Check GPU status
    print("\nGPU Status (nvidia-smi):")
    gpu_info = get_nvidia_smi_info()
    if gpu_info:
        print(gpu_info["output"])
    else:
        print("  nvidia-smi not found or no GPU detected")


if __name__ == "__main__":
    test_cuda()