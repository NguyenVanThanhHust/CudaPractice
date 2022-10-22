#include <cuda_runtime.h>
#include <string>
#include <iostream>


using std::cin;
using std::cout;
using std::endl;


__global__ 
void vectorAdd(float *A, float *B, float *C, int numElements)
{
    int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadIndex < numElements)
    {
        C[threadIndex] = A[threadIndex] + B[threadIndex];
    }
      
}

int main()
{
    int numElements=2000;
    cudaError_t err = cudaSuccess;
    size_t size = numElements*sizeof(float);

    // Host vector
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        cout<<"Failed to allocate host vector  "<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = (float)(i+1);
        h_B[i] = (float)(i+2);
    }
    
    float *d_A=NULL, *d_B=NULL, *d_C=NULL;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess)
    {
        cout<<"Failed to allocate device vector for vector A  "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess)
    {
        cout<<"Failed to allocate device vector for vector B  "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess)
    {
        cout<<"Failed to allocate device vector for vector C  "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cout<<"Failed to copy from host vector A to device vector A "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cout<<"Failed to copy from host vector B to device vector B "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        cout<<"Failed to launch kernel "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cout<<"Failed to copy from device C to host C "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            float local_value = h_A[i] + h_B[i];
            cout<<"Failed at "<<i<<" expect: "<<local_value<<" get: "<<h_C[i]<<endl;
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    cout<<"Test passed"<<endl;

    // Free device global memory
    err = cudaFree(d_A);
    if (err != cudaSuccess)
    {
        cout<<"Failed to free data on device A "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);
    if (err != cudaSuccess)
    {
        cout<<"Failed to free data on device B "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);
    if (err != cudaSuccess)
    {
        cout<<"Failed to free data on device C "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }

    free(h_A);
    free(h_B);
    free(h_C);
       
    return 0;
}