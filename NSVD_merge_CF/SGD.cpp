#include "SGD.h"
#include <iostream>
using namespace std;

void SGD::gradientDescent(Predictor& predictor, int max_iterations, int step_length, float alpha, float lambda, float epsilon) {
    std::cout << "start gradient descent" << std::endl;

    float training_rmse = 99999999.0;
    float validation_rmse = 99999999.0;
    float prev_validation_rmse = 99999999.0;
    float prev_training_rmse = 99999999.0;

    for (int iter = 0; iter < max_iterations; ++ iter) {
        clock_t start, finish;
        start = clock();

        training_rmse = predictor.train(step_length, alpha, lambda);
        validation_rmse = predictor.validation(step_length);
        alpha *= 0.99;

        finish = clock();
        float duration =  (float)(finish - start) / CLOCKS_PER_SEC;
        std::cout << "iter: " << iter 
            << " training RMSE: " << training_rmse << " validation RMSE: " << validation_rmse 
            << ", time=" << duration << std::endl; 

        if (training_rmse + validation_rmse + epsilon > prev_training_rmse + prev_validation_rmse) {
            break;
        } else {
            prev_training_rmse = training_rmse;
            prev_validation_rmse = validation_rmse;
        }
    } // end for iter

    printf("gradient descent is finished\n");
}

