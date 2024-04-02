cmake_minimum_required(VERSION 3.2)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
option(USE_CUDA "Use CUDA" ON)

enable_language("CUDA")
add_definitions(-DUSE_CUDA)

add_executable(asyncAPI asyncAPI.cu)

include_directories(asyncAPI
  ../../Common/
  /usr/local/cuda/include
)

link_directories(asyncAPI
  /usr/local/cuda/lib64/
)

target_link_libraries(asyncAPI
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
)
