// includes, system
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/utsname.h>
#endif
#include <iostream>
#include <cstring>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // helper utility functions

#ifndef MAX
#define MAX(a, b) (a>b ? a:b)
#endif

using std::cin;
using std::cout;
using std::endl;


const char *sampleName = "simpleAssert";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////
//! Tests assert function.
//! Thread whose id > N will print assertion failed error message.
////////////////////////////////////////////////////////////////////////////////
__global__ void testKernel(int N) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    assert(gtid < N);
}

void runTest(int args, char **cudaDevAttrGpuOverlap)
{
    int Nblocks = 2;
    int Nthreads = 32;
    cudaError_t error;

    #ifndef _WIN32
        utsname OS_System_Type;
        uname(&OS_System_Type);

        printf("OS_System_Type.release = %s\n", OS_System_Type.release);

        if (!strcasecmp(OS_System_Type.sysname, "Darwin")) {
            printf("simpleAssert is not current supported on Mac OSX\n\n");
            exit(EXIT_SUCCESS);
        } else {
            printf("OS Info: <%s>\n\n", OS_System_Type.version);
        }

    #endif
}

int main(int argc, char *argv[])
{
    printf("%s starting...\n", sampleName);
    

    return EXIT_SUCCESS;
}