#include "sigmoid.h"

__device__ float sigmoid(float x){
    return 1.0f / (1+exp(-x));
};

__global__ void sigmoidActivationForward(float *input, float *output, int input_dim_x, int input_dim_y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_dim_x * input_dim_y)
    {
        output[index] = sigmoid(input[index]);
    }
};

__global__ void sigmoidActivattionBackward(float *)