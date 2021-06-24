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
void verify_result(int *a, int *b, int *c, int N)
{
    for(int i=0; i < N; i++)
    {
        assert(c[i] == a[i] + b[i]);
    }
}

// Main function
int main()
{
    constexpr int NUM_ELEM = 2048;
    constexpr size_t array_size = sizeof(int)*NUM_ELEM;
    // Allocate unified memory
    int *a, *b, *c;
    
    checkCudaErrors(cudaMalloc(&a, array_size));
    checkCudaErrors(cudaMalloc(&b, array_size));
    checkCudaErrors(cudaMalloc(&c, array_size));
    
    // Initialize array
    for(int i=0; i<NUM_ELEM; i++)
    {
        a[i] = rand() %100;
        b[i] = rand() %100;
    }


    int NUM_THREADS = 1024;
    int NUM_BLOCKS = (NUM_ELEM + NUM_THREADS - 1)/ NUM_THREADS;

    // Launch the kernel on the GPU
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c, NUM_ELEM);

    // Synchronize from device
    cudaDeviceSynchronize();

    // Check the result
    verify_result(a, b, c, NUM_ELEM);

    // Free CUDA memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    std::cout<<"COMPLETE!!!"; 
    return 0;

}
