import numpy as np
from numba import cuda
import time
from typing import Tuple


def cpu_square_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix * matrix


@cuda.jit
def gpu_square_matrix(matrix_in: np.ndarray, matrix_out: np.ndarray) -> None:
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if row < matrix_in.shape[0] and col < matrix_in.shape[1]:
        matrix_out[row, col] = matrix_in[row, col] * matrix_in[row, col]


def test_cuda_operation(size: int = 1000) -> Tuple[float, float]:
    matrix = np.random.random((size, size)).astype(np.float32)
    matrix_gpu = cuda.to_device(matrix)
    output_gpu = cuda.device_array_like(matrix_gpu)

    threadsperblock = (16, 16)
    blockspergrid_x = (matrix.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (matrix.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start_gpu = time.time()
    gpu_square_matrix[blockspergrid, threadsperblock](matrix_gpu, output_gpu)
    cuda.synchronize()
    gpu_result = output_gpu.copy_to_host()
    gpu_time = time.time() - start_gpu

    start_cpu = time.time()
    cpu_result = cpu_square_matrix(matrix)
    cpu_time = time.time() - start_cpu

    if np.allclose(cpu_result, gpu_result):
        print("\nCUDA Test Results:")
        print("✓ CUDA operation successful")
        print("✓ Results match CPU computation")
        print(f"CPU time: {cpu_time:.4f} seconds")
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        return gpu_time, cpu_time
    else:
        print("✗ Results don't match - CUDA may not be working correctly")
        return 0, 0


if __name__ == "__main__":
    if not cuda.is_available():
        print("CUDA is not available on your system")
    else:
        print("CUDA is available")
        device = cuda.get_current_device()
        print(f"\nCUDA Device Information:")
        print(f"Name: {device.name.decode()}")
        print(f"Max threads per block: {device.max_threads_per_block}")
        print(f"Max block dimensions: ({device.max_block_dim_x}, {device.max_block_dim_y}, {device.max_block_dim_z})")
        print(f"Compute Capability: {device.compute_capability}")

        test_cuda_operation()