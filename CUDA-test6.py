import cv2
import sys
import platform
from cv2 import cuda

def verify_cuda():
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    cuda.printCudaDeviceInfo(0)

    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"\nCUDA devices found: {count}")

        if count > 0:
            device = cv2.cuda.getDevice()
            print(f"Current device ID: {device}")

            # Test basic GPU operation
            test_mat = cv2.cuda_GpuMat()
            print("\nGPU Mat creation successful")

            # Get device properties
            print("\nDevice Properties:")
            print(f"Compute capability: {cv2.cuda.getDevice().computeCapability()}")
            print(f"Total memory: {cv2.cuda.getDevice().totalMemory() / (1024 * 1024):.2f} MB")
            print(f"Max threads per block: {cv2.cuda.getDevice().maxThreadsPerBlock}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPossible solutions:")
        print("1. Uninstall current OpenCV:")
        print("   pip uninstall opencv-python opencv-python-headless opencv-contrib-python")
        print("2. Install CUDA-enabled OpenCV:")
        print("   pip install opencv-python-cuda==4.8.0.76")
        print("3. Ensure NVIDIA drivers are up to date")
        print("4. Verify CUDA installation using nvidia-smi command")


if __name__ == "__main__":
    verify_cuda()