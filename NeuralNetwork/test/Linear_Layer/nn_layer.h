#pragma once
#include <iostream>
#include <string>

#include "matrix.h"

class NNLayer{
protected:
    std::string name;

public:
    virtual ~NNLayer() = 0;

    virtual Matrix& forward(Matrix& output)=0;
    virtual Matrix& backward(Matrix& d_input, float lr) = 0;

    std::string getName() {return this->name; };
};