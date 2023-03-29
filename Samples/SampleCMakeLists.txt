cmake_minimum_required(VERSION 3.2)
project(deepstream)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -pthread")

find_package(OpenCV REQUIRED)

add_executable(main 
    src/data/mnist.cpp
    src/data/mnist.hpp
    src/data/mnist.cu
    src/main.cpp
    )

message("OpenCV include dir")
message(${OpenCV_INCLUDE_DIRS})

include_directories(main
  /usr/local/cuda-11.8/include
  /usr/include/gstreamer-1.0
  /usr/include/glib-2.0
  /usr/lib/x86_64-linux-gnu/glib-2.0/include
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(
  /usr/local/cuda-11.8/lib64/
  /usr/local/cuda/lib64/
)

target_link_libraries(main
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
  ${OpenCV_LIBS}
)       