#include <cuda_runtime.h>
#include <iostream>
#include "helper_cuda.h"
#include "add_kernel.cuh"

__global__ void add(int* a, int* b, int* c, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void addKernelCuda(int* a, int* b, int* c, int n)
{
    int *d_a, *d_b, *d_c; 
    size_t size = n * sizeof(int); 
    // Allocate device memory 
    checkCudaErrors(cudaMalloc((void**)&d_a, size)); 
    checkCudaErrors(cudaMalloc((void**)&d_b, size)); 
    checkCudaErrors(cudaMalloc((void**)&d_c, size)); 
    
    // Copy inputs to device 
    checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice)); 
    checkCudaErrors(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice)); 
    
    // Launch kernel with 256 threads per block and n/256 blocks 
    add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n); 
    
    // Copy result back to host 
    checkCudaErrors(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost)); 
    // Free device memory 
    checkCudaErrors(cudaFree(d_a)); 
    checkCudaErrors(cudaFree(d_b)); 
    checkCudaErrors(cudaFree(d_c));
}