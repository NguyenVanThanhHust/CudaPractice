cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)
 
include(CTest)
 
add_library(mat_mul STATIC
    device_array.h
    kernel.cu
    kernel.h 
    
  )
 
# Request that mat_mul be built with -std=c++11
# As this is a public compile feature anything that links to 
# mat_mul will also build with -std=c++11
target_compile_features(mat_mul PUBLIC cxx_std_11)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( mat_mul
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
add_executable(particle_test main.cu)
 
set_property(TARGET particle_test 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(particle_test PRIVATE mat_mul)
 