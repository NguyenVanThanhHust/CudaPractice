cmake_minimum_required(VERSION 3.2)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
option(USE_CUDA "Use CUDA" ON)

enable_language("CUDA")
add_definitions(-DUSE_CUDA)

add_executable(vectorAddRT vectorAddRT.cu)

include_directories(vectorAddRT
  ../../Common/
  /usr/local/cuda/include
)

link_directories(vectorAddRT
  /usr/local/cuda/lib64/
)

target_link_libraries(vectorAddRT
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
)
