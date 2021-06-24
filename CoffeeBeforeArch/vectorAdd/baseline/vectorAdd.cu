#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <algorithm>
#include <cassert>
#include <array>
#include <cstdlib>
#include <iostream>

// CUDA kernle for vector addition
__global__ void vectorAdd(int *a, int *b,int *c, int n)
{
    // Calculate global thread id 
    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(thread_id < n)
    {
        c[thread_id] = a[thread_id] + b[thread_id];
    }
}

//Check vector add result
void verify_result(std::array<int, 2048> &a, std::array<int, 2048> &b, std::array<int, 2048> &c)
{
    for(int i=0; i < a.size(); i++)
    {
        assert(c[i] == a[i] + b[i]);
    }
}

// Main function
int main()
{
    constexpr int NUM_ELEM = 2048;
    constexpr size_t array_size = sizeof(int)*NUM_ELEM;
    // Allocate memory on host
    std::array<int, NUM_ELEM> h_a, h_b, h_c;
    for(int i=0; i<NUM_ELEM; i++)
    {
        h_a[i] = rand() %100;
        h_b[i] = rand() %100;
    }
    
    // Allocate memory on device/gpu
    int *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc(&d_a, array_size));
    checkCudaErrors(cudaMalloc(&d_b, array_size));
    checkCudaErrors(cudaMalloc(&d_c, array_size));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a.data(), array_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_b, h_b.data(), array_size, cudaMemcpyHostToDevice);

    int NUM_THREADS = 1024;
    int NUM_BLOCKS = (NUM_ELEM + NUM_THREADS - 1)/ NUM_THREADS;

    // Launch the kernel on the GPU
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, NUM_ELEM);

    // Copy from device to host
    cudaMemcpy(h_c.data(), d_c, array_size, cudaMemcpyDeviceToHost);
    // Check the result
    verify_result(h_a, h_b, h_c);

    // Free CUDA memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout<<"COMPLETE!!!"; 
    return 0;

}
