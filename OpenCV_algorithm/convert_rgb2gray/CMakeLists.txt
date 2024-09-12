cmake_minimum_required(VERSION 3.2)
project(deepstream)

set(CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS " -O3 -pthread")

find_package(OpenCV REQUIRED)

add_executable(main 
    main.cpp
    )

add_executable(create_fake_image 
create_fake_image.cpp
    )

message("OpenCV include dir")
message(${OpenCV_INCLUDE_DIRS})

include_directories(main
  /usr/local/cuda/include
  ../../Common
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(
  /usr/local/cuda/lib64/
)

target_link_libraries(main
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
  ${OpenCV_LIBS}
)       

include_directories(create_fake_image
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(create_fake_image
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
  ${OpenCV_LIBS}
)  
