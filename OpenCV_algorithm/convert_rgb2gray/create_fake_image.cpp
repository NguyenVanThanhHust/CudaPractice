#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    int height = 5;
    int width = 10;

    cv::Mat mat(height, width, CV_8UC3);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            mat.at<cv::Vec3b>(y, x)[0] = 1;
            mat.at<cv::Vec3b>(y, x)[1] = 2;
            mat.at<cv::Vec3b>(y, x)[2] = 3;
        }
    }
    cv::imwrite("sample.jpg", mat);
    return 0;    
}