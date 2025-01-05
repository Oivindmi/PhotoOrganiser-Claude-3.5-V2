import os
import ctypes
import sys


def check_system_cuda():
    system32_path = os.path.join(os.environ['SystemRoot'], 'System32')
    print(f"Checking System32: {system32_path}")

    try:
        nvcuda = ctypes.CDLL(os.path.join(system32_path, 'nvcuda.dll'))
        print("✓ nvcuda.dll found in System32 and loadable")
    except Exception as e:
        print(f"✗ Error loading nvcuda.dll: {e}")

    # Check NVIDIA driver
    try:
        import subprocess
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode()
        print("\nNVIDIA Driver Info:")
        print(nvidia_smi)
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")

    # Try CUDA import
    try:
        import cv2
        print(f"\nOpenCV version: {cv2.__version__}")
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"CUDA devices detected: {device_count}")

        if device_count > 0:
            # Test basic CUDA operation
            test_mat = cv2.cuda_GpuMat()
            print("✓ CUDA GpuMat creation successful")
    except Exception as e:
        print(f"Error with OpenCV CUDA: {e}")


if __name__ == "__main__":
    check_system_cuda()