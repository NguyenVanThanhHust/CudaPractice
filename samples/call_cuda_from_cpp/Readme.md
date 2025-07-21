# Call CUDA from CPP
This is an example to demonstrate how to write a kernel funtion in CUDA, compile it to lib then call it from c++

## How to build
```
cmake -S . -B build && cmake --build build
```

In folder `build`, you should see
```
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
detect_cuda_compute_capabilities.cu
libcpu_add.so*
libcuda_add_kernel.so*
main_program*
Makefile
```

## How to run
To run on GPU
```
DEVICE=CUDA ./main_program
```

To run on CPU
```
DEVICE=CPU ./main_program
```