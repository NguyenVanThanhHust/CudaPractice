cmake_minimum_required(VERSION 3.2)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
option(USE_CUDA "Use CUDA" ON)

enable_language("CUDA")
add_definitions(-DUSE_CUDA)

add_executable(cpp11_cuda 
  range.hpp
  c++11_cuda.cu
)

include_directories(cpp11_cuda
  ../../Common/
  /usr/local/cuda/include
)

link_directories(cpp11_cuda
  /usr/local/cuda/lib64/
)

target_link_libraries(cpp11_cuda
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
)
