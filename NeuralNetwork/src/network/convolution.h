#pragma once
#include <vector>
#include <string>

class Convolution
{
public:
    int k1, k2, in_channel, out_channel;

    std::string mode="cpu";

    Convolution(int in_channel, int out_channel, int k1, int k2);

    void forward();
    void backward();
    void forward_cpu();
    void backward_cpu();
    void forward_gpu();
    void backward_gpu();
};