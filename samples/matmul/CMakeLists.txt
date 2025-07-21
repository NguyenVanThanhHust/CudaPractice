cmake_minimum_required(VERSION 3.18)
project(matmul LANGUAGES CXX CUDA)

# Optionally, link against CUDA libraries
find_package(CUDAToolkit REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
message("$(CUDA_ARCH_LIST)")


# Specify the CUDA source file 
set(CUDA_SRC 
    helper_cuda.h
    helper_string.h
    matmul.cuh 
    matmul.cu) 

# Create the CUDA shared library 
add_library(cuda_matmul_kernel SHARED ${CUDA_SRC})

# Specify the C++ source file for executable progra
set(EXE_PROG_SRC main.cpp) 
# Create the executable from the C++ source file 
add_executable(main_program ${EXE_PROG_SRC}) 
# set_property(TARGET main_program PROPERTY "${CUDA_ARCH_LIST}")
set_property(TARGET main_program PROPERTY CUDA_ARCHITECTURES "70;75;86")

# Set CUDA properties
set_target_properties(main_program PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link the executable against the CUDA shared library 
target_link_libraries(main_program cuda_matmul_kernel ${CUDA_CUDART_LIBRARY} CUDA::cudart)