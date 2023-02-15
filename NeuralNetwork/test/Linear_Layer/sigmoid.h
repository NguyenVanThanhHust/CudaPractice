#pragma once
#include "nn_layer.h"

class Sigmoid : public NNLayer {
private:
    Matrix output;
    Matrix input;
    Matrix d_input;

public: 
    Sigmoid(std::string name);
    ~Sigmoid();

    Matrix& forward(Matrix& input);
    Matrix& backward(Matrix& d_output, float learning=0.1);
};

