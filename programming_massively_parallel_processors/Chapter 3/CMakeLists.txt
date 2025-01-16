ARG CUDA_VERSION

cmake_minimum_required(VERSION 3.2)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
option(USE_CUDA "Use CUDA" ON)

enable_language("CUDA")
add_definitions(-DUSE_CUDA)

add_executable(simpleVectorAdd simpleVectorAdd.cu)

include_directories(simpleVectorAdd
  ../app-common/includes/
  ../includes/
  /usr/local/cuda-12.3/include
  /usr/include/glib-2.0
  /usr/lib/x86_64-linux-gnu/glib-2.0/include
)

link_directories(
  /usr/local/cuda-12.3/lib64/
  /usr/local/cuda/lib64/
)

target_link_libraries(simpleVectorAdd
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
)
