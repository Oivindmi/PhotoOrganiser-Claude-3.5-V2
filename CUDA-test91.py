import os
import ctypes
import sys


def check_cuda_dlls():
    cuda_path = os.environ.get('CUDA_PATH')
    print(f"CUDA_PATH: {cuda_path}")

    if cuda_path:
        bin_path = os.path.join(cuda_path, 'bin')
        print(f"\nChecking {bin_path} for CUDA DLLs...")

        # Required DLLs
        dlls = ['cudart64_12.dll', 'nvcuda.dll', 'cublas64_12.dll']

        for dll in dlls:
            dll_path = os.path.join(bin_path, dll)
            if os.path.exists(dll_path):
                try:
                    ctypes.CDLL(dll_path)
                    print(f"✓ {dll}: Found and loadable")
                except Exception as e:
                    print(f"✗ {dll}: Found but not loadable - {e}")
            else:
                print(f"✗ {dll}: Not found")

        # Check if bin path is in PATH
        system_path = os.environ.get('PATH', '').split(os.pathsep)
        if bin_path.lower() in [p.lower() for p in system_path]:
            print(f"\n✓ CUDA bin path is in system PATH")
        else:
            print(f"\n✗ CUDA bin path is NOT in system PATH!")
            print("Add this to your system PATH:")
            print(bin_path)


if __name__ == "__main__":
    check_cuda_dlls()