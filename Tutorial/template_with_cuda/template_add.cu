#include "template_add.cuh"
#include "helper_cuda.h"

namespace SampleNamespace {

    // CUDA kernel definition
    template <typename T>
    __global__ void addKernel(const T* a, const T* b, T* c, int n) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < n) {
            c[index] = a[index] + b[index];
        }
    }

    // Template function definition
    template <typename T>
    void add(const T* a, const T* b, T* c, int n) {
        T *d_a, *d_b, *d_c;
        size_t size = n * sizeof(T);

        // Allocate device memory
        checkCudaErrors(cudaMalloc((void**)&d_a, size));
        checkCudaErrors(cudaMalloc((void**)&d_b, size));
        checkCudaErrors(cudaMalloc((void**)&d_c, size));

        // Copy inputs to device
        checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

        // Launch kernel with 256 threads per block
        addKernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

        // Copy result back to host
        checkCudaErrors(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

        // Free device memory
        checkCudaErrors(cudaFree(d_a));
        checkCudaErrors(cudaFree(d_b));
        checkCudaErrors(cudaFree(d_c));
    }

    // Explicit template instantiations
    template void add<int>(const int* a, const int* b, int* c, int n);
    template void add<float>(const float* a, const float* b, float* c, int n);
    template void add<__half>(const __half* a, const __half* b, __half* c, int n);
    template void add<__nv_bfloat16>(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int n);

} // namespace SampleNamespace
