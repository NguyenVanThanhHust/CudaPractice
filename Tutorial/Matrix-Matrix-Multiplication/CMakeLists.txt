cmake_minimum_required(VERSION 3.15)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(mat_mul LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 20)

option(USE_CUDA "Use CUDA" ON)

add_executable(mat_mul 
    # main.cu
    main_simple.cu
    )

include_directories(
    /usr/local/cuda/include
    /usr/local/cuda/lib64/
    /usr/local/cuda/lib64/
    )

target_link_libraries(mat_mul 
    ${CUDA_LIBRARIES}
    /usr/local/cuda/lib64/libcudart.so
    )

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA) 
endif()