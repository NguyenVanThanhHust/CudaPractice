cmake_minimum_required(VERSION 3.2)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
option(USE_CUDA "Use CUDA" ON)

enable_language("CUDA")
add_definitions(-DUSE_CUDA)

add_executable(clock clock.cu)

include_directories(clock
  ../../Common/
  /usr/local/cuda-12.0/include
)

link_directories(clock
  /usr/local/cuda-12.0/lib64/
  /usr/local/cuda/lib64/
)

target_link_libraries(clock
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
)
