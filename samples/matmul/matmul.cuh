#ifndef TEMPLATE_ADD_KERNEL_H
#define TEMPLATE_ADD_KERNEL_H

#include <cuda_runtime.h>
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace MatmulNamespace {

    // Template function declaration
    template <typename T>
    void matmul(const T* host_a, const T* host_b, T* host_c, int m, int k, int n);

    // CUDA kernel declaration
    template <typename T>
    __global__ void matmulKernel(const T* A, const T* B, T* C, int M, int K, int N);

} // namespace MatmulNamespace
#endif // TEMPLATE_ADD_KERNEL_H
