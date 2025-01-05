import sys
import numpy as np
from typing import Dict, Any


def test_pytorch() -> Dict[str, Any]:
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        device_name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else "N/A"
        cuda_version = torch.version.cuda if cuda_available else "N/A"

        return {
            "available": cuda_available,
            "device_count": device_count,
            "device_name": device_name,
            "version": cuda_version
        }
    except ImportError:
        return {"error": "PyTorch not installed"}


def test_tensorflow() -> Dict[str, Any]:
    try:
        import tensorflow as tf
        cuda_available = len(tf.config.list_physical_devices('GPU')) > 0
        gpu_devices = tf.config.list_physical_devices('GPU')
        device_count = len(gpu_devices)

        return {
            "available": cuda_available,
            "device_count": device_count,
            "devices": [device.name for device in gpu_devices] if device_count > 0 else []
        }
    except ImportError:
        return {"error": "TensorFlow not installed"}


def test_cuda() -> None:
    print("\nTesting CUDA Availability\n" + "=" * 50)

    # Test PyTorch
    print("\nPyTorch CUDA Status:")
    pytorch_status = test_pytorch()
    if "error" in pytorch_status:
        print(f"  {pytorch_status['error']}")
    else:
        print(f"  CUDA Available: {pytorch_status['available']}")
        print(f"  Device Count: {pytorch_status['device_count']}")
        print(f"  Device Name: {pytorch_status['device_name']}")
        print(f"  CUDA Version: {pytorch_status['version']}")

    # Test TensorFlow
    print("\nTensorFlow CUDA Status:")
    tf_status = test_tensorflow()
    if "error" in tf_status:
        print(f"  {tf_status['error']}")
    else:
        print(f"  CUDA Available: {tf_status['available']}")
        print(f"  Device Count: {tf_status['device_count']}")
        if tf_status['device_count'] > 0:
            print("  Devices Found:")
            for device in tf_status['devices']:
                print(f"    - {device}")

    # Test NumPy
    print("\nNumPy Version:")
    print(f"  {np.__version__}")


if __name__ == "__main__":
    test_cuda()