#include "template_add.cuh"

int main() {
    const int n = 10;
    int a[n], b[n], c[n];
    float af[n], bf[n], cf[n];

    // Initialize input arrays
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = i * 2;
        af[i] = static_cast<float>(i);
        bf[i] = static_cast<float>(i * 2);
    }

    // Call the CUDA template function for integers
    SampleNamespace::add(a, b, c, n);

    // Print the result for integers
    std::cout << "Result (int): ";
    for (int i = 0; i < n; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Call the CUDA template function for floats
    SampleNamespace::add(af, bf, cf, n);

    // Print the result for floats
    std::cout << "Result (float): ";
    for (int i = 0; i < n; ++i) {
        std::cout << cf[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
