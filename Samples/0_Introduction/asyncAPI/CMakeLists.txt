cmake_minimum_required(VERSION 3.2)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
option(USE_CUDA "Use CUDA" ON)

enable_language("CUDA")
add_definitions(-DUSE_CUDA)

add_executable(asyncAPI asyncAPI.cu)

include_directories(asyncAPI
  ../../Common/
  /usr/local/cuda-11.2/include
  /usr/include/gstreamer-1.0
  /usr/include/glib-2.0
  /usr/lib/x86_64-linux-gnu/glib-2.0/include
)

link_directories(asyncAPI
  /usr/local/cuda-11.2/lib64/
  /usr/local/cuda/lib64/
)

target_link_libraries(asyncAPI
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
)
