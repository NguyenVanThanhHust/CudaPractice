#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"


// Initialize a vector
void init_vector(float *a, int n)
{
    for(int i=0; i<n; i++)
    {
        a[i] = (float)(rand() % 100);
    }
}

// Verify the result
void verify_result(float *a, float *b, float *c, float factor, int n)
{
    for(int i=0; i<n; i++)
    {
        assert(c[i] == factor*a[i]+b[i]);
    }
}

int main()
{
    // Vector size
	int n = 1 << 16;
	size_t bytes = n * sizeof(float);
	
	// Declare vector pointers
	float *h_a, *h_b, *h_c;
	float *d_a, *d_b;

	// Allocate memory
	h_a = (float*)malloc(bytes);
	h_b = (float*)malloc(bytes);
	h_c = (float*)malloc(bytes);
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);

	// Initialize vectors
	init_vector(h_a, n);
	init_vector(h_b, n);

    // Create and initialize a new context
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // Copy the vector to the device
    cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1); // 1: step size
    cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);

    // Launch simple sxapy kernel single precision a*x + y
    const float scale = 2.0f;
    cublasSaxpy(handle, n, &scale, d_a, 1, d_b, 1);
    
    // Copy the result to host device
    cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);

    verify_result(h_a, h_b, h_c, scale, n);

    // Clean up the vreated handle
    cublasDestroy(handle);

    // Release allocated memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    std::cout<<"Successull execuation"<<std::endl;
    return 0;
}