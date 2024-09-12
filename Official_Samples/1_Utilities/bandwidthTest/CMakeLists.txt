# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.18)

# Set the project name
project(bandwidthTest LANGUAGES CXX CUDA)
set (CMAKE_CXX_STANDARD 14)

# Find the CUDA toolkit
find_package(CUDA REQUIRED)

# Set the CUDA compiler flags
set(CUDA_NVCC_FLAGS
    -O3
    -arch=compute_87
    -code=sm_87
    )

# Include common helper functions
include_directories(../../Common)

# Add the CUDA kernel file
cuda_add_executable(bandwidthTest bandwidthTest.cu)

# Install the executable
# install(TARGETS bandwidthTest DESTINATION bin)