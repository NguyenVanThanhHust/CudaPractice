#include <cuda_runtime.h>
#include "bgr2gray.cuh"
#include "helper_cuda.h"

__global__ void bgr_to_gray(const unsigned char* bgr, unsigned char* gray, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * channels;
    int gray_idx = y * width + x;

    if (x < width && y < height) {
        unsigned char b = bgr[idx];
        unsigned char g = bgr[idx + 1];
        unsigned char r = bgr[idx + 2];
        gray[gray_idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}
void convert_bgr_2_gray_gpu(unsigned char *bgr_image, unsigned char* gray_image, int height, int width)
{
    unsigned char *d_bgr_image, *d_gray_image; 
    size_t array_size = height * width * sizeof(unsigned char);
    // Allocate device memory 
    checkCudaErrors(cudaMalloc((void**)&d_bgr_image, array_size*3)); 
    checkCudaErrors(cudaMalloc((void**)&d_gray_image, array_size)); 
    
    // Copy inputs to device 
    checkCudaErrors(cudaMemcpy(d_bgr_image, bgr_image, array_size*3, cudaMemcpyHostToDevice)); 
    checkCudaErrors(cudaMemcpy(d_gray_image, gray_image, array_size, cudaMemcpyHostToDevice)); 
    
    // Define block and grid sizes
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    bgr_to_gray<<<grid, block>>>(d_bgr_image, d_gray_image, width, height, 3);

    // Copy result back to host 
    checkCudaErrors(cudaMemcpy(gray_image, d_gray_image, array_size, cudaMemcpyDeviceToHost)); 
    // Free device memory 
    checkCudaErrors(cudaFree(d_bgr_image)); 
    checkCudaErrors(cudaFree(d_gray_image)); 
}
