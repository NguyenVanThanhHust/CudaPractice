#include "bgr2gray.h"

void convert_bgr_2_gray_cpu(unsigned char *bgr_image, unsigned char* gray_image, int height, int width)
{
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            int location = i + j*width;
            unsigned char b = bgr_image[location*3];
            unsigned char g = bgr_image[location*3+1];
            unsigned char r = bgr_image[location*3+2];
            gray_image[location] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
}