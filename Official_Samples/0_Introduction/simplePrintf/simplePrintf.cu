// includes, system
#include <iostream>
#include <cstring>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // helper utility functions

#ifndef MAX
#define MAX(a, b) (a>b ? a:b)
#endif

using std::cin;
using std::cout;
using std::endl;


__global__ void testKernel(int val)
{
    // gridDim.x*blockIdx.y+blockIdx.x is block location in grid
    // gridDim.x in number of block in each row
    // blockIdx.y is number of row 
    printf("[Block id: %d thread Id: %d]: Value is: %d \n", gridDim.x*blockIdx.y+blockIdx.x, threadIdx.z*blockDim.y*blockDim.x+threadIdx.y*blockDim.x+threadIdx.x, val);
}

int main(int argc, char *argv[])
{
    int devID;
    cudaDeviceProp deviceProps;

    cout<<"Starting..."<<argv[0]<<endl;

    // Pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);
    
    // get GPU information
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, deviceProps.name, deviceProps.major, deviceProps.minor);

    printf("printf() is called. Output: \n \n");
    // Kernel config, create 2 dim grid, each location 
    // have 3-dim block
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    testKernel<<<dimGrid, dimBlock>>>(10);
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}