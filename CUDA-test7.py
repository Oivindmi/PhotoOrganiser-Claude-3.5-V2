import cv2
import platform

import torch
print(torch.cuda.is_available())



def check_opencv_cuda():
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Platform: {platform.platform()}")

    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"\nCUDA devices detected: {count}")

        if count > 0:
            # Test basic CUDA functionality
            mat = cv2.cuda_GpuMat()
            print("✓ CUDA GpuMat creation successful")
            return True
    except Exception as e:
        print(f"\nError: {e}")
        print("\nCUDA support might not be available in this OpenCV build.")
        print("Try reinstalling OpenCV with:")
        print("pip install --upgrade opencv-contrib-python")
    return False


if __name__ == "__main__":
    check_opencv_cuda()