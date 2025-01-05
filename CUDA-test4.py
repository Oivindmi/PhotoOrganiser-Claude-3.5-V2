import subprocess
import os
from typing import Dict, Optional, List, Tuple


def find_cuda_paths() -> List[str]:
    base_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    cuda_paths = []
    if os.path.exists(base_path):
        for folder in os.listdir(base_path):
            if folder.startswith('v'):
                full_path = os.path.join(base_path, folder)
                if os.path.isdir(full_path):
                    cuda_paths.append(full_path)
    return sorted(cuda_paths)


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

    # Check CUDA paths
    env_cuda_path = os.environ.get('CUDA_PATH')
    env_cuda_home = os.environ.get('CUDA_HOME')
    actual_cuda_paths = find_cuda_paths()

    print("\nCUDA Paths:")
    print(f"  Environment CUDA_PATH: {env_cuda_path or 'Not set'}")
    print(f"  Environment CUDA_HOME: {env_cuda_home or 'Not set'}")
    print("\nInstalled CUDA versions found on system:")
    if actual_cuda_paths:
        for path in actual_cuda_paths:
            print(f"  {path}")
    else:
        print("  No CUDA installations found in default location")

    if env_cuda_path and env_cuda_path not in actual_cuda_paths:
        print("\nWARNING: Environment CUDA_PATH points to a different location than detected installations!")

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