#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <algorithm>
#include <cassert>
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// #include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
// #include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include "bgr2gray.h"
#include "bgr2gray.cuh"
#include "utils.h"

using std::cout;
using std::endl;

int main(int argc, char** argv)
{
    if ( argc != 2 )
    {
        printf("usage: main <Image_Path>\n");
        return -1;
    }

    // Load the input image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    int height = image.rows;
    int width = image.cols;
    int num_channels = image.channels();

    unsigned char* bgr_array = new unsigned char[height * width * num_channels]; 
    // Copy data from cv::Mat to the array 
    std::memcpy(bgr_array, image.data, height * width * num_channels * sizeof(unsigned char));

    unsigned char* gray_array = new unsigned char[height * width]; 
    convert_bgr_2_gray_cpu(bgr_array, gray_array, height, width);

    cv::Mat gray_image(height, width, CV_8UC1, gray_array);
    cv::imwrite("gray_image_cpu.png", gray_image);

    unsigned char* gpu_gray_array = new unsigned char[height * width]; 
    convert_bgr_2_gray_gpu(bgr_array, gpu_gray_array, height, width);

    cv::Mat gpu_gray_image(height, width, CV_8UC1, gpu_gray_array);
    cv::imwrite("gray_image_gpu.png", gpu_gray_image);
    return 0;
}