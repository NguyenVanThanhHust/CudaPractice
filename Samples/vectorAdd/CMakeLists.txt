cmake_minimum_required(VERSION 3.25)
project(vectorAdd LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
option(USE_CUDA "Use CUDA" ON)

find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME} 
    vectorAdd.cu
    helper_cuda.h
    helper_string.h
    )

link_directories(${PROJECT_NAME} 
    /usr/local/cuda/lib64/
)
target_link_libraries(${PROJECT_NAME} 
    CUDA::cudart
    ${CUDA_LIBRARIES}
    /usr/local/cuda/lib64/libcudart.so
)