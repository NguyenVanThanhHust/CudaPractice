cmake_minimum_required(VERSION 3.10)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

option(USE_CUDA "Use CUDA" ON)

enable_language("CUDA")
add_definitions(-DUSE_CUDA)

add_executable(cppOverload cppOverload_kernel.cuh cppOverload.cu)
set_property(TARGET cppOverload PROPERTY CUDA_ARCHITECTURES native)
include_directories(cppOverload
  ../../Common/
  /usr/local/cuda/include
)

link_directories(cppOverload
  /usr/local/cuda/lib64/
)

target_link_libraries(cppOverload
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
)
