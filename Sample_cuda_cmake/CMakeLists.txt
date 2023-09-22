cmake_minimum_required(VERSION 3.25)
project(myapp)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} CUDA::cudart)