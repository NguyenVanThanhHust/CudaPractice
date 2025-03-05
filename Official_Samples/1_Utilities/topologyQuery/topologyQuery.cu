
/*
 * This is a simple test program to measure the memcopy bandwidth of the GPU.
 * It can measure device to device copy bandwidth, host to device copy bandwidth
 * for pageable and pinned memory, and device to host copy bandwidth for
 * pageable and pinned memory.
 *
 * Usage:
 * ./bandwidthTest [option]...
 */

// CUDA runtime
#include <cuda_runtime.h>

// includes
#include <helper_cuda.h>  // helper functions for CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

#include <cuda.h>

#include <cassert>
#include <iostream>
#include <memory>

static const char *sSDKsample = "CUDA Bandwidth Test";

int main(int argc, char **argv) {
  int deviceCount = 0;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Enumerates Device <-> Device links
  for (int device1 = 0; device1 < deviceCount; device1++) {
    for (int device2 = 0; device2 < deviceCount; device2++) {
      if (device1 == device2) continue;

      int perfRank = 0;
      int atomicSupported = 0;
      int accessSupported = 0;

      checkCudaErrors(cudaDeviceGetP2PAttribute(
          &accessSupported, cudaDevP2PAttrAccessSupported, device1, device2));
      checkCudaErrors(cudaDeviceGetP2PAttribute(
          &perfRank, cudaDevP2PAttrPerformanceRank, device1, device2));
      checkCudaErrors(cudaDeviceGetP2PAttribute(
          &atomicSupported, cudaDevP2PAttrNativeAtomicSupported, device1,
          device2));

      if (accessSupported) {
        std::cout << "GPU" << device1 << " <-> GPU" << device2 << ":"
                  << std::endl;
        std::cout << "  * Atomic Supported: "
                  << (atomicSupported ? "yes" : "no") << std::endl;
        std::cout << "  * Perf Rank: " << perfRank << std::endl;
      }
    }
  }

  // Enumerates Device <-> Host links
  for (int device = 0; device < deviceCount; device++) {
    int atomicSupported = 0;
    checkCudaErrors(cudaDeviceGetAttribute(
        &atomicSupported, cudaDevAttrHostNativeAtomicSupported, device));
    std::cout << "GPU" << device << " <-> CPU:" << std::endl;
    std::cout << "  * Atomic Supported: " << (atomicSupported ? "yes" : "no")
              << std::endl;
  }

  return 0;
}