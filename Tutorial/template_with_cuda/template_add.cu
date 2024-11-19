#include "template_add.cuh"

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
        cudaMalloc((void**)&d_a, size);
        cudaMalloc((void**)&d_b, size);
        cudaMalloc((void**)&d_c, size);

        // Copy inputs to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        // Launch kernel with 256 threads per block
        addKernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

        // Copy result back to host
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    // Explicit template instantiations
    template void add<int>(const int* a, const int* b, int* c, int n);
    template void add<float>(const float* a, const float* b, float* c, int n);

} // namespace SampleNamespace
