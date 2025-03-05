
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

// define for project
// defines, project
#define MEMCOPY_ITERATIONS 100
#define DEFAULT_SIZE (32 * (1e6))      // 32 M
#define DEFAULT_INCREMENT (4 * (1e6))  // 4 M
#define CACHE_CLEAR_SIZE (16 * (1e6))  // 16 M

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

// enums
enum testMode {QUICK_MODE, RANGE_MODE, SHMOO_MODE};
enum memcpyKind {DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE};
enum printMode { USER_READABLE, CSV};
enum memoryMode { PINNED, PAGEABLE};

const char *sMemoryCopyKind[] = {"Device to Host", "Host to Device",
                                 "Device to Device", NULL};

const char *sMemoryMode[] = {"PINNED", "PAGEABLE", NULL};

// if true, use CPU based timing for everything
static bool bDontUseGPUTiming;

int *pArgc = NULL;
char **pArgv = NULL;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(const int argc, const char **argv);
void testBandwidth(unsigned int start, unsigned int end, unsigned int increment);

void printHelp(void);

int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    // allocate buffer in cpu
    flush_buf = (char*)malloc(FLUSH_SIZE);

    // set logfile name and start to logs
    printf("[%s] - Starting...\n", sSDKsample);

    int returnVal = runTest(argc, (const char**)argv);

    return 0;
}

int runTest(const int argc, const char **argv) {
    int start = DEFAULT_SIZE;
    int end = DEFAULT_SIZE;
    int startDevice = 0;
    int endDevice = 0;
    int increment = DEFAULT_INCREMENT;
    testMode = QUICK_MODE;
    bool htod = false;
    bool dtoh = false;
    bool dtod = false;
    bool wc = false;
    char *modeStr;
    char *device = NULL;
    printMode printmode = USER_READABLE;
    char *memModeStr = NULL;
    memoryMode memMode = PINNED;
    
    // process command line args
    if (checkCmdLineFlag(argc, argv, "help")) {
        printHelp();
        return 0;
    }
    if (checkCmdLineFlag(argc, argv, "csv"))
    {
        printMode=CSV;
    }
    if (getCmdLineArgumentString(argc, argv, "memory", &memoryModeStr))
    {
        /* code */
    }
    
    

}

void printHelp(void) {
    printf("Usage: bandwidthTest [OPTION]... \n");
    printf("Test teh bandwidth for device to host, host to device, device to device transfers \n");
    printf("\n");

    printf(
        "Example: measure for device to host pinned memory copies"
        " in the range 1024 bytes to 10240 bytes in 1024 byte incrementes \n"
    );
    printf(
        "./bandwidthTest --memory=pinned --mode=range --start=1024 --end=102400 "
        "--increment=1024 --dtoh\n"
    );

    printf("\n");
    printf("Options:\n");
    printf("--help\tDisplay this help menu\n");
    printf("--csv\tPrint results as a CSV\n");
    printf("--device=[deviceno]\tSpecify the device device to be used\n");
    printf("  all - compute cumulative bandwidth on all the devices\n");
    printf("  0,1,2,...,n - Specify any particular device to be used\n");
    printf("--memory=[MEMMODE]\tSpecify which memory mode to use\n");
    printf("  pageable - pageable memory\n");
    printf("  pinned   - non-pageable system memory\n");
    printf("--mode=[MODE]\tSpecify the mode to use\n");
    printf("  quick - performs a quick measurement\n");
    printf("  range - measures a user-specified range of values\n");
    printf("  shmoo - performs an intense shmoo of a large range of values\n");

    printf("--htod\tMeasure host to device transfers\n");
    printf("--dtoh\tMeasure device to host transfers\n");
    printf("--dtod\tMeasure device to device transfers\n");
    #if CUDART_VERSION >= 2020
        printf("--wc\tAllocate pinned memory as write-combined\n");
    #endif
        printf("--cputiming\tForce CPU-based timing always\n");

    printf("Range mode options\n");
    printf("--start=[SIZE]\tStarting transfer size in bytes\n");
    printf("--end=[SIZE]\tEnding transfer size in bytes\n");
    printf("--increment=[SIZE]\tIncrement size in bytes\n");
}