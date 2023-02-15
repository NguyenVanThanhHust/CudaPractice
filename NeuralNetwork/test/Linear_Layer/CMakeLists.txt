cmake_minimum_required(VERSION 3.2)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set_property(PROPERTY CXX_STANDARD 17)
option(USE_CUDA "Use CUDA" ON)

enable_language("CUDA")
add_definitions(-DUSE_CUDA)

add_executable(LinearLayer main.cpp)

include_directories(LinearLayer
  ../../Common/
  /usr/local/cuda-12.0/include
)

link_directories(LinearLayer
  /usr/local/cuda-12.0/lib64/
  /usr/local/cuda/lib64/
)

target_link_libraries(LinearLayer
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
)