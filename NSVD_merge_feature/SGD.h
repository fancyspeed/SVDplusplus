#pragma once
#include <stdio.h>
#include "Predictor.h"

class SGD {
  public:
    SGD() { }
    virtual ~SGD() { }

    virtual void gradientDescent(Predictor& predictor, int max_iterations, int step_length, float alpha, float lambda, float epsilon);
};
