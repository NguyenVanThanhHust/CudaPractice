#ifndef _DEVICE_ARRAY_H_
#define _DEVICE_ARRAY_H_

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

template <class T>
class device_array
{
    public:
        explicit device_array()
        : start_(0), end_(0)
        {}

        // constructor
        explicit device_array(size_t size)
        {
            allocate(size);
        }

        // destructor
        ~device_array()
        {
            free;
        }

        // resize vector
        void resize(size_t size)
        {
            free();
            allocate(size);
        }

        // Get size of the array
        size_t getSize() const
        {
            return end_ - start_;
        }

        // get Data
        const T* getData() const
        {
            return start_;
        }

        T* getData()
        {
            return start_;
        }

        // set
        void set(const T* src, size_t size)
        {
            size_t min = std::min(size, getSize());
            cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
            if (result != cudaSuccess)
            {
                throw std::runtime_error("failed to copy to device memory");
            }
        }
        // get
        void get(T* dest, size_t size)
        {
            size_t min = std::min(size, getSize());
            cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
            if (result != cudaSuccess)
            {
                throw std::runtime_error("failed to copy to host memory");
            }
        }
    private:
    void allocate(size_t size)
    {
        cudaError_t  result = cudaMalloc((void**)&start_, size * sizeof(T));
        if (result != CUDA_SUCCESS)
        {
            start_ = end_ = 0;
            throw std::runtime_error("Failed to allocate memory");
        }
        end_ = start_ + date;
    }

    void free()
    {
        if (start_ != 0)
        {
            cudaFree(start_);
            start_ = end_ = 0;
        }
    }

    T* start_;
    T* end_;
}


#endif