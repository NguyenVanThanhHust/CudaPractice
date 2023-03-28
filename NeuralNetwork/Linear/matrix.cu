#include "matrix.h"
#include <memory>

// https://stackoverflow.com/questions/13061979/shared-ptr-to-an-array-should-it-be-used
template< typename T >
struct host_array_deleter
{
    void operator ()( T const * p)
    { 
        delete[] p; 
    }
};

template< typename T >
struct device_array_deleter
{
    void operator ()( T const * p)
    { 
        cudaFree[] p; 
    }
};

Matrix::Matrix(size_t dim_x, size_t dim_y) :
    shape(dim_x, dim_y), data_device(nullptr), data_host(nullptr),
    device_allocated(false), host_allocated(false);
{}

Matrix::Matrix(Shape shape):
    Matrix(shape.x, shape.y)
{}

void Matrix::allocateHostMemory()
{
    if (!host_allocated)
    {
        // https://stackoverflow.com/questions/13061979/shared-ptr-to-an-array-should-it-be-used
        data_host = std::shared_ptr<float>(new float[shape.x*shape.y], host_array_deleter);
        host_allocated = true;
    }
}

void Matrix::allocateDeviceMemory()
{
    if (!device_allocated)
    {
        float *device_memory = nullptr;
        cudaMalloc(&device_memory, shape.x*shape.y*sizeof(float));
        NNException::throwIfDeviceErrorsOccurred("Can't allocate CUDA Memory for Tensor");
        data_device = std::shared_ptr<float>(device_memory, device_array_deleter);
        device_allocated = true;
    }
}

void Matrix::allocateMemory()
{
    allocateHostMemory();
    allocateDeviceMemory();
}

void Matrix::allocateMemoryIfNotAllocated()
{
    if (!device_allocated && !host_allocated)
    {
        allocateHostMemory();
        allocateDeviceMemory();
        host_allocated = true;
        device_allocated = true;
    }
}

void Matrix::copyHostToDevice()
{
    if (!host_allocated)
    {
        throw NNException("Host memory isn't allocated");
    };
    if (!device_allocated)
    {
        throw NNException("Device memory isn't allocated");
    };
    cudaMemcpy(data_device.get(), data_host.get(), shape.x*shape.y*sizeof(float), cudaMemcpyHostToDevice);
    NNException::throwIfDeviceErrorsOccurred("Can't copy data from host to device");
}

void Matrix::copyDeviceToHost()
{
    if (!host_allocated)
    {
        throw NNException("Host memory isn't allocated");
    };
    if (!device_allocated)
    {
        throw NNException("Device memory isn't allocated");
    };
    cudaMemcpy(data_device.get(), data_host.get(), shape.x*shape.y*sizeof(float), cudaMemcpyDeviceToHost);
    NNException::throwIfDeviceErrorsOccurred("Can't copy data from device to host");
}

float& Matrix::operator[](const int index)
{
    return data_host.get()[index];
}

const float& Matrix::operator[](const int index) const 
{
    return data_host.get()[index];
}