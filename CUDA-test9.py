import cv2
import sys
import os


def verify_opencv():
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV path: {cv2.__file__}")
    print(f"\nCUDA support:")

    # Check CUDA modules
    cuda_modules = [attr for attr in dir(cv2.cuda) if not attr.startswith('_')]
    print(f"CUDA modules available: {len(cuda_modules) > 0}")

    # Try CUDA operation
    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"CUDA devices: {gpu_count}")
        if gpu_count > 0:
            # Test GPU Mat creation
            mat = cv2.cuda_GpuMat()
            print("✓ CUDA is functional")
    except Exception as e:
        print(f"✗ CUDA error: {str(e)}")

    # Additional system info
    print(f"\nPython version: {sys.version}")
    cuda_path = os.environ.get('CUDA_PATH')
    print(f"CUDA_PATH: {cuda_path}")


if __name__ == "__main__":
    verify_opencv()