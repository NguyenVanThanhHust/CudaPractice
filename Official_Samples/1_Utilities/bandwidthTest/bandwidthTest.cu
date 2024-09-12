//CUDA runtime
#include <cuda_runtime.h>

// includes
#include <helper_cuda.h>
#include <helper_functions.h>

#include <cuda.h>
#include <cassert>
#include <iostream>
#include <memory>

// Defines project
#define MEMCOPY_ITERATIONS 100
#define DEFAULT_SIZE (32*(1e6))       // 32 M
#define DEFAULT_INCREMENT (4*(1e6))   // 4 M
#define CACHE_CLEAR_SIZE (16*(1e6))   // 16 M

// shmoo mode defines
#define SHMOO_MEMSIZE_MAX (64 * (1e6))       // 64 M
#define SHMOO_MEMSIZE_START (1e3)            // 1 KB
#define SHMOO_INCREMENT_1KB (1e3)            // 1 KB
#define SHMOO_INCREMENT_2KB (2 * 1e3)        // 2 KB
#define SHMOO_INCREMENT_10KB (10 * (1e3))    // 10KB
#define SHMOO_INCREMENT_100KB (100 * (1e3))  // 100 KB
#define SHMOO_INCREMENT_1MB (1e6)            // 1 MB
#define SHMOO_INCREMENT_2MB (2 * 1e6)        // 2 MB
#define SHMOO_INCREMENT_4MB (4 * 1e6)        // 4 MB
#define SHMOO_LIMIT_20KB (20 * (1e3))        // 20 KB
#define SHMOO_LIMIT_50KB (50 * (1e3))        // 50 KB
#define SHMOO_LIMIT_100KB (100 * (1e3))      // 100 KB
#define SHMOO_LIMIT_1MB (1e6)                // 1 MB
#define SHMOO_LIMIT_16MB (16 * 1e6)          // 16 MB
#define SHMOO_LIMIT_32MB (32 * 1e6)          // 32 MB

// CPU cache flush
#define FLUSH_SIZE (256 * 1024 * 1024)
char *flush_buf;

// enums, project
enum testMode {QUICK_MODE, RANGE_MODE, SHMOP_MODE};
enum memcpyKind {DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE};
enum printMode {USER_READABLE, CSV};
enum memoryMode {PINNED, PAGEABLE};

const char *sMemoryCopyKind[] = {"Device to Host", "Host to Device",
                                 "Device to Device", NULL};

const char *sMemoryMode[] = {"PINNED", "PAGEABLE", NULL};

// if true, use CPU based timing for everything
static bool bDontUseGPUTiming;

int *pArgc = NULL;
char **pArgv = NULL;

int runTest(const int argc, const char **argv);
void testBandwidth(unsigned int start, unsigned int end, unsigned int increment,
                testMode mode, memcpyKind kind, printMode printmode,
                memoryMode memMode, int startDevice, int endDevice, bool wc);

void testBandwidthWidth(unsigned int size, memcpyKind kind, printMode printmode, 
                memoryMode memMode, int startDevice, int endDevice, bool wc);


int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    flush_buf = (char*)malloc(FLUSH_SIZE);
}

