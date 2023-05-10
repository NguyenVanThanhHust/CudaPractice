cmake_minimum_required(VERSION 3.15)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(mat_mul LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 20)

option(USE_CUDA "Use CUDA" ON)

add_executable(matmul 
    matmul.cu
    )

add_executable(matmul_simple 
    matmul_simple.cu
    )

include_directories(
    /usr/local/cuda/include
    /usr/local/cuda/lib64/
    )

target_link_libraries(matmul 
    ${CUDA_LIBRARIES}
    /usr/local/cuda/lib64/libcudart.so
    )

target_link_libraries(matmul_simple 
    ${CUDA_LIBRARIES}
    /usr/local/cuda/lib64/libcudart.so
    )


message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()