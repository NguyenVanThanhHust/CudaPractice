#pragma once
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "data/mnist.hpp"
#include "Constants.h"

int main()
{
    std::string data_path = TRAIN_DATA_PATH;
    thrust::host_vector<thrust::host_vector<float>> train_data;
    read_data(data_path=TRAIN_DATA_PATH, train_data);

    std::string data_path = TEST_DATA_PATH;
    thrust::host_vector<thrust::host_vector<float>> test_data;
    read_data(data_path=TEST_DATA_PATH, test_data);

    return 0;
}