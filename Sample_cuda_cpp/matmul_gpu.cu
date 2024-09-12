#include "matmul_gpu.cuh"
#include "helper_cuda.h"

__global__ void matmul_kernel(int *a, int *b, int *c, int m, int n, int p)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    while (row < m && col < p)
    {
        int temp_value = 0;
        for (int i = 0; i < n; i++)
        {
            temp_value += a[row*n + i] * b[i*p+col];
        }
        c[row*p + col] = temp_value;
    }
}

void matmul(int *host_a, int *host_b, int *host_c, int m, int n, int p)
{
    int BLOCK_SIZE=16;
    int *dev_a, *dev_b, *dev_c;
    size_t size_a = sizeof(int)*m*n;
    size_t size_b = sizeof(int)*n*p;
    size_t size_c = sizeof(int)*m*p;
    checkCudaErrors(cudaMalloc(&dev_a, size_a));
    checkCudaErrors(cudaMalloc(&dev_b, size_b));
    checkCudaErrors(cudaMalloc(&dev_c, size_c));

    cudaMemcpy(dev_a, host_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size_b, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (p + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    matmul_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, m, n, p);
    
}
