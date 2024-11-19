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

using std::cout;
using std::endl;

std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
    }

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

    // Get the number of rows (height) and columns (width)
    int height = image.rows;
    int width = image.cols;

    int mat_type = image.type();
    cout<<"height: "<< height<<endl;
    cout<<"width: "<< width<<endl;
    cout<<"type: "<<type2str(mat_type)<<endl;

    // Convert the image to a vector
    std::vector<uchar> image_vector;
    if (image.isContinuous()) {
        // If the image is stored in a continuous block of memory
        image_vector.assign(image.datastart, image.dataend);
    } else {
        // If the image is not stored in a continuous block of memory
        for (int i = 0; i < image.rows; ++i) {
        image_vector.insert(image_vector.end(), image.ptr<uchar>(i), image.ptr<uchar>(i) + image.cols * image.channels());
        }
    }
    std::vector<int> int_image_vector(image_vector.begin(), image_vector.end());
    cout<<int_image_vector.size()<<endl;
    for (int i = 0; i < 100; i++)
    {
        cout<<(int)int_image_vector[i]<<" ";
    }
    cout<<(int)int_image_vector[100]<<" "<<(int)int_image_vector[100+height*width]<<" "<<(int)int_image_vector[100+2*height*width]<<endl;
    
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // Convert the image to a vector
    std::vector<uchar> gray_image_vector;
    if (gray_image.isContinuous()) {
        // If the image is stored in a continuous block of memory
        gray_image_vector.assign(gray_image.datastart, gray_image.dataend);
    } else {
        // If the image is not stored in a continuous block of memory
        for (int i = 0; i < gray_image.rows; ++i) {
        gray_image_vector.insert(gray_image_vector.end(), gray_image.ptr<uchar>(i), gray_image.ptr<uchar>(i) + gray_image.cols * gray_image.channels());
        }
    }
    std::vector<int> int_gray_image_vector(gray_image_vector.begin(), gray_image_vector.end());
    cout<<int_gray_image_vector.size()<<endl;
    for (int i = 0; i < 50; i++)
    {
        cout<<(int)int_gray_image_vector[i]<<" ";
    }
    cout<<(int)int_gray_image_vector[100]<<endl;
    
    return 0;
}