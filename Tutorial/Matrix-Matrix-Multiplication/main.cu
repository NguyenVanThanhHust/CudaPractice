#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cassert>
#include <cmath>

#include <cuda_runtime.h>

#define assertm(exp, msg) assert(((void)msg, exp))

using std::cout;
using std::endl;


__global__ void matMulKernel(float *d_A, float *d_B, float *d_C, int height_A, int width_A, int height_B, int width_B)
{
    int row_threadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int col_threadIdx = blockDim.y * blockIdx.y + threadIdx.y;
    if(row_threadIdx < height_A && col_threadIdx)
    {
        float temp_value = 0;
        for(int i=0; i < width_A; i++)
        {
            temp_value += d_A[row_threadIdx * width_A + i] * d_B[i * width_B + col_threadIdx];
        }
        d_C[row_threadIdx * width_B + col_threadIdx] = temp_value;
    }
};

int main(int argc, char **argv)
{
    int height_A, width_A, height_B, width_B;
    height_A = std::stoi(argv[1]);
    width_A = std::stoi(argv[2]);
    height_B = std::stoi(argv[3]);
    width_B = std::stoi(argv[4]);
    assertm(width_A==height_B, "height matrix A must equal to width of matrix B");
    
    // Declare host array
    float *h_A, *h_B, *h_C, *cpu_C;

    // Initialize host 
    h_A = new float[height_A*width_A];
    h_B = new float[height_B*width_B];
    h_C = new float[height_A*width_B];
    cpu_C = new float[height_A*width_B];

    for(int row_index=0; row_index < height_A; row_index++){
        for(int col_index=0; col_index < width_A; col_index++){
            h_A[col_index + row_index*width_A] = (float)sin(col_index + row_index*width_A);
        }
    }

    for(int row_index=0; row_index < height_B; row_index++){
        for(int col_index=0; col_index < width_B; col_index++){
            h_B[col_index + row_index*width_B] = (float)cos(col_index + row_index*width_B);
        }
    }

    for(int row_index=0; row_index < height_A; row_index++){
        for(int col_index=0; col_index < width_B; col_index++){
            float temp_C = 0;
            for(int temp_index=0; temp_index < width_A; temp_index++){
                temp_C += h_A[row_index * width_A + temp_index] * h_B[temp_index * width_B + col_index];
            } 
            cpu_C[row_index * width_B + col_index] = temp_C;
        }
    }

    cudaError_t err = cudaSuccess;

    // Declare device array
    float *d_A=NULL, *d_B=NULL, *d_C=NULL;
    int sizeA = height_A * width_A * sizeof(float);
    err = cudaMalloc((void **)&d_A, sizeA);
    if (err != cudaSuccess)
    {
        std::cout<<"Failed to allocate device vector for vector A  "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<std::endl;
        exit(EXIT_FAILURE);
    }


    int sizeB = height_B * width_B * sizeof(float);
    err = cudaMalloc((void **)&d_B, sizeB);
    if (err != cudaSuccess)
    {
        std::cout<<"Failed to allocate device vector for vector B  "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<std::endl;
        exit(EXIT_FAILURE);
    }

    int sizeC = height_A * width_B * sizeof(float);
    err = cudaMalloc((void **)&d_C, sizeC);
    if (err != cudaSuccess)
    {
        std::cout<<"Failed to allocate device vector for vector C  "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<std::endl;
        exit(EXIT_FAILURE);
    }

    for(int row_index=0; row_index < height_A; row_index++){
        for(int col_index=0; col_index < width_B; col_index++){
            std::cout<< cpu_C[row_index * width_B + col_index] << " ";
        }
        std::cout<<std::endl;
    }

    err = cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cout<<"Failed to copy from host vector A to device vector A "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cout<<"Failed to copy from host vector B to device vector B "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid(1, 1, 1);
    matMulKernel<<<threadsPerBlock, blocksPerGrid>>>(d_A, d_B, d_C, height_A, width_A, height_B, width_B);   
    cudaDeviceSynchronize();
    cudaError_t result = cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        std::cout<<"Failed to copy from device to host vector C  "<<cudaGetErrorString(err)<<" at file: "<<__FILE__<<" at line: "<<__LINE__<<std::endl;
        exit(EXIT_FAILURE);
    }

    for(int row_index=0; row_index < height_A; row_index++){
        for(int col_index=0; col_index < width_B; col_index++){
            cout<<h_C[row_index * width_B + col_index]<< " ";
        }
        std::cout<<std::endl;
    }

    for(int row_index=0; row_index < height_A; row_index++){
        for(int col_index=0; col_index < width_B; col_index++){
            if (fabs(cpu_C[row_index * width_B + col_index] - h_C[row_index * width_B + col_index]) > 1e-5){
                cout<<"Failed at row "<<row_index<<" col: "<<col_index;
                cout<<" expect: "<<cpu_C[row_index * width_B + col_index]<<" get: "<<h_C[row_index * width_B + col_index]<<endl;
                exit(EXIT_FAILURE);
            }
        }
        std::cout<<std::endl;
    }
    cout<<"Test passed"<<endl;

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

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] cpu_C;

    return 0;
}