#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "matmul_cpu.h"
#include "matmul_gpu.cuh"

using std::cout;
using std::endl;

int main(int argc, char** argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Num device: " << deviceCount << std::endl;

    int m, n, p;
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    p = std::atoi(argv[3]);
    cout<<"m: "<<m<<" n: "<<n<<" p: "<<p<<endl;

    // cout<<"Initialize vector"<<endl;
    // std::vector<float> matrixA, matrixB, matrixC;
    // for (int i = 0; i < m*n; i++)
    // {
    //     float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    //     matrixA.push_back(r);
    // }
    // for (int i = 0; i < n*p; i++)
    // {
    //     float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    //     matrixB.push_back(r);
    // }

    cout<<"Initialize vector"<<endl;
    std::vector<int> matrixA, matrixB, matrixC;
    matrixA.reserve(m*n);
    matrixB.reserve(n*p);
    matrixC.reserve(m*p);
    for (int i = 0; i < m*n; i++)
    {
        matrixA.push_back(i+1);
    }
    for (int i = 0; i < n*p; i++)
    {
        matrixB.push_back(i+2);
    }
    std::cout << "matrixA.size() = " << matrixA.size() << '\n';  // Will output 3
    std::cout << "matrixB.size() = " << matrixB.size() << '\n';  // Will output 3

    cout<<"Initialize array"<<endl;
    int h_A[m*m], h_B[n*p], h_C[m*p];
    std::copy(matrixA.begin(), matrixA.end(), h_A);
    std::copy(matrixB.begin(), matrixB.end(), h_B);
    
    cout<<"Start to mat mul on cpu"<<endl;
    runOnCPU(h_A, h_B, h_C, m, n, p);
    
    matrixC.insert(matrixC.begin(), h_C + 0, h_C + m*p);
    cout<<"Print result"<<endl;
    for(int x : matrixC)
    {
        cout<<x<<" ";
    }
    cout<<endl;
    
    cout<<"Copy data to gpu array"<<endl;

    cout<<"Start to mat mul on gpu"<<endl;
    

    return 0;
}