#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "helper_cuda.h"

using std::cout;
using std::generate;
using std::vector;

// Static shared mem calculation for convenience (Int 16x16 matrix)
const int N = 1 << 10;
const int SHMEM_SIZE = 1 << 10;


__global__ void tiledMatrixMul(const int *a, const int *b, int *c) {
    // Two static size pieces of shared memory
    __shared__ int share_a[SHMEM_SIZE];
    __shared__ int share_b[SHMEM_SIZE];

    // Shorten these parameters for clean re use
    // Calculate global row and column positions for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_val = 0;
    // Sweep tiles over entire matrix
    for(int i = 0; i < N; i += blockDim.x)
    // step is a block dim, move each share m
    {
        /*
        Every thread in a thread block loads one elemnt into shared memeory
        The element location in shared memory corresponds to the thread's position
        in the thread block 

        */
        // Load elemtn for this tile:
        share_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row*N + i + threadIdx.x];
        share_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i*N + threadIdx.y*N + col];

        // Wait for both tiles to be loaded before doing computation
        __syncthreads();

        // Do matrix multiplication on the small matrix
        for(int j=0; j < blockDim.x; j++)
        {
            temp_val += share_a[threadIdx.y * blockDim.x + j] * share_b[j * blockDim.x + threadIdx.x];
        }
        __syncthreads();
        // Write back the results
        c[row * N + col] = temp_val;
    }
}

// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

int main() {

    // Size (in bytes) of matrix
    size_t bytes = N * N * sizeof(int);

    // Host vectors
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);

    // Initialize matrices
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per CTA dimension
    int THREADS = 32;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = N / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);


    // Launch kernel
    tiledMatrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result
    verify_result(h_a, h_b, h_c);

    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}   