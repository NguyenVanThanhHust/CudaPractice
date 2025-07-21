#ifndef TEMPLATE_ADD_KERNEL_H
#define TEMPLATE_ADD_KERNEL_H

#include <cuda_runtime.h>
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace AddNamespace {

    // Template function declaration
    template <typename T>
    void add(const T* a, const T* b, T* c, int n);

    // CUDA kernel declaration
    template <typename T>
    __global__ void addKernel(const T* a, const T* b, T* c, int n);

} // namespace AddNamespace
#endif // TEMPLATE_ADD_KERNEL_H
