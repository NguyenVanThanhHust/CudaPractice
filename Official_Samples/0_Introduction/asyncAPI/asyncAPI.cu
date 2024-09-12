// includes, system
#include <iostream>
#include <cstring>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // helper utility functions

using std::cin;
using std::cout;
using std::endl;

__global__ void increment_kernel(int *g_data, int inc_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}

bool correct_output(int *data, const int n, const int x)
{
    for (int i = 0; i < n; i++)
    {
        if (data[i]!=x)
        {
            // cout<<"This is wrong at "i<<" data"<<data[i]<<" must be"<<x<<endl;
            printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
            return false;
        }
        return true;
    }
}


int main(int argc, char *argv[])
{
    int devID;
    cudaDeviceProp deviceProps;

    cout<<"Starting..."<<argv[0]<<endl;

    // Pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);
    
    // get device name
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    cout<<"CUDA device "<<deviceProps.name;

    // Create 3d array 
    int n = 16*1024*1024;
    int nbytes = n * sizeof(int);
    int value;

    // allocate host memory
    int *a = 0;
    checkCudaErrors(cudaMalloc((void **)&a, nbytes));
    memset(a, 0, nbytes);

    // allocate device memory
    int *d_a = 0;
    checkCudaErrors(cudaMalloc((void **)&a, nbytes));
    checkCudaErrors(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
    dim3 threads = dim3(512, 1, 1);
    dim3 blocks = dim3(n/threads.x, 1, 1);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface *timer=NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    // asynchronously issue to work to the GPU
    checkCudaErrors(cudaProfilerStart());
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    checkCudaErrors(cudaProfilerStop());

    // have CPU to do some work while waiting to finish
    unsigned long int counter=0;
    while (cudaEventQuery(stop)==cudaErrorNotReady)
    {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n",
            counter);

    // check the output for correctness
    bool bFinalResults = correct_output(a, n, value);

    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFree(d_a));

    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}