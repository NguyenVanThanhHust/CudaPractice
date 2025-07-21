#include "template_add.cuh"

int main() {
    const int n = 10;
    int a[n], b[n], c[n];
    float af[n], bf[n], cf[n];
    __half af16[n], bf16[n], cf16[n];
 
    // Initialize input arrays
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = i * 2;
        af[i] = static_cast<float>(i);
        bf[i] = static_cast<float>(i * 2);
        af16[i] = __float2half(af[i]);
        bf16[i] = __float2half(bf[i]);
    }

    // Call the CUDA template function for integers
    AddNamespace::add(a, b, c, n);

    // Print the result for integers
    std::cout << "Result (int): ";
    for (int i = 0; i < n; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Call the CUDA template function for floats
    AddNamespace::add(af, bf, cf, n);

    // Print the result for floats
    std::cout << "Result (float): ";
    for (int i = 0; i < n; ++i) {
        std::cout << cf[i] << " ";
    }
    std::cout << std::endl;

    // Call the CUDA template function for float16
    AddNamespace::add(af16, bf16, cf16, n);

    // Print the result for float 16
    std::cout << "Result (float 16): ";
    for (int i = 0; i < n; ++i) {
        std::cout << __half2float(cf16[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}
