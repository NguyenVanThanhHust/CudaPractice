#pragma once
#include "shape.hpp"
#include <memory>

class Matrix {
private:
    bool device_allocated;
    bool host_allocated;

    void allocateDeviceMemory();
    void allocateHostMemory();

public:
    Shape shape;

    // Create pointer to save data
    std::shared_ptr<float> data_device;
    std::shared_ptr<float> data_host;

    // Initialize by shape or explicity dim
    Matrix(Shape shape);
    Matrix(size_t dim_x, size_t dim_y);

    // allocate memory on host and device
    void allocateMemory();
    void allocateMemoryIfNotAllocated();
    void copyHostToDevice();
    void copyDeviceToHost();

    // get value at specific position
    float& operator[](const int index);
    const float& operator[](const int index) const;
};