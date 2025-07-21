#include <cstdlib>
#include <iostream>
#include <string>
#include "add_kernel.cuh"
#include "add_function.h"

int main() {
    const int n = 10;
    int a[n], b[n], c[n];

    // Initialize input arrays
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }
    const char* ENV_DEVICE = std::getenv("DEVICE");
    const char* default_device = "cpu";
    const char* DEVICE = ENV_DEVICE ? ENV_DEVICE : default_device;
    std::string GPU_DEVICE = "CUDA";
    std::cout<<DEVICE<<std::endl;
    if (GPU_DEVICE.compare(DEVICE) == 0)
    {
        std::cout<<"run on gpu"<<std::endl;
        // Call the CUDA function
        addKernelCuda(a, b, c, n);
    }
    else
    {
        std::cout<<"run on cpu"<<std::endl;
        addCpu(a, b, c, n);
    }

    // Print the result
    std::cout << "Result: ";
    for (int i = 0; i < n; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
