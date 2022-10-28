#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using std::cin;
using std::cout;
using std::endl;

void read_data(std::string data_path, thrust::host_vector<thrust::host_vector<float>> &data); 
