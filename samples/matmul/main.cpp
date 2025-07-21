#include <iostream>
#include "matmul.cuh"

using std::cout;
using std::endl;

int main() {
    const int m=2, n=3, k=4;
    int a[m*k], b[k*n], c[m*n];
    float af[m*k], bf[k*n], cf[m*n];
    __half af16[m*k], bf16[k*n], cf16[m*n];
 
    // Initialize input arrays
    for (int i = 0; i < m*k; ++i) {
        a[i] = i;
        af[i] = static_cast<float>(i);
        af16[i] = __float2half(af[i]);
        cout<<a[i] <<" ";
    }
    cout<<endl<<" ";

    for (int i = 0; i < k*n; ++i) {
        b[i] = i ;
        bf[i] = static_cast<float>(i);
        bf16[i] = __float2half(bf[i]);
        cout<<b[i] <<" ";
    }
    cout<<endl<<" ";

    // Call the CUDA template function for integers
    MatmulNamespace::matmul(a, b, c, m, k, n);

    // Print the result for integers
    std::cout << "Result (int): ";
    for (int i = 0; i < m*n; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Call the CUDA template function for floats
    MatmulNamespace::matmul(af, bf, cf, m, k, n);

    // Print the result for floats
    std::cout << "Result (float): ";
    for (int i = 0; i < m*n; ++i) {
        std::cout << cf[i] << " ";
    }
    std::cout << std::endl;

    // // Call the CUDA template function for float16
    // MatmulNamespace::matmul(af16, bf16, cf16, m, k, n);

    // // Print the result for float 16
    // std::cout << "Result (float 16): ";
    // for (int i = 0; i < m*n; ++i) {
    //     std::cout << __half2float(cf16[i]) << " ";
    // }
    // std::cout << std::endl;

    return 0;
}
