#include "matmul.cuh"
#include "helper_cuda.h"

#include <iostream>
using std::endl;
using std::cout;

namespace MatmulNamespace {

    // CUDA kernel definition for matrix multiplication
    // A is M x K
    // B is K x N
    // C is M x N
    template <typename T>
    __global__ void matmulKernel(const T* A, const T* B, T* C, int M, int K, int N) {
        // Calculate the row and column of the current thread for the output matrix C
        int row = blockIdx.y * blockDim.y + threadIdx.y; // Row in C (and A)
        int col = blockIdx.x * blockDim.x + threadIdx.x; // Column in C (and B)

        float sum = 0.0f;

        // Check if the current thread is within the bounds of the output matrix C
        if (row < M && col < N) {
            // Perform the dot product for C[row][col]
            // Iterate through the columns of A (which is K) or rows of B (which is K)
            for (int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }

    // Template function definition
    template <typename T>
    void matmul(const T* host_A, const T* host_B, T* host_C, int M, int K, int N) {
        T *device_A, *device_B, *device_C;
        size_t size_A = M*K * sizeof(T);
        size_t size_B = K*N * sizeof(T);
        size_t size_C = M*N * sizeof(T);
        // cout<<size_A<<" "<<size_B<<" "<<size_C<<" "<<endl;

        // Allocate device memory
        checkCudaErrors(cudaMalloc((void**)&device_A, size_A));
        checkCudaErrors(cudaMalloc((void**)&device_B, size_B));
        checkCudaErrors(cudaMalloc((void**)&device_C, size_C));

        // Copy inputs to device
        checkCudaErrors(cudaMemcpy(device_A, host_A, size_A, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_B, host_B, size_B, cudaMemcpyHostToDevice));

        int threadsPerBlockX = 16; // Threads along N dimension (columns of C)
        int threadsPerBlockY = 16; // Threads along M dimension (rows of C)

        dim3 blockDim(threadsPerBlockX, threadsPerBlockY);

        // Calculate grid dimensions based on C's M x N dimensions
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

        matmulKernel<<<gridDim, blockDim>>>(device_A, device_B, device_C, M, K, N);
        cudaDeviceSynchronize();

        // Copy result back to host
        checkCudaErrors(cudaMemcpy(host_C, device_C, size_C, cudaMemcpyDeviceToHost));

        // Free device memory
        checkCudaErrors(cudaFree(device_A));
        checkCudaErrors(cudaFree(device_B));
        checkCudaErrors(cudaFree(device_C));
    }

    // Explicit template instantiations
    template void matmul<int>(const int* host_A, const int* host_B, int* host_C, int M, int K, int N);
    template void matmul<float>(const float* host_A, const float* host_B, float* host_C, int M, int K, int N);
    // template void matmul<__half>(const __half* host_A, const __half* host_B, __half* host_C, int M, int K, int N);
    // template void matmul<__nv_bfloat16>(const __nv_bfloat16* host_A, const __nv_bfloat16* host_B, __nv_bfloat16* host_C, int M, int K, int N);

} // namespace MatmulNamespace
