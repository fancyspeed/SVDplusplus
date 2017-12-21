#include "Predictor.h"
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
using namespace std;

float Predictor::train(int step_length, float alpha, float lambda) {
    printf("start to train, alpha:%f, lambda:%f\n", alpha, lambda);

    float sum_rmse = 0;
    srand( time(NULL) );
    int line_idx = 0;
    int line_process = 0;
    int size_tot = (int) train_log.size();

    while (true) {
        line_idx += rand() % step_length + 1;
        if (line_idx >= size_tot) break;

        // !!! modify !!! for MAP
        sum_rmse += update(line_idx, alpha, lambda) / user_activity[train_log[line_idx].user_idx];

        ++ line_process;
    }

    printf("after training, sum_rmse:%f, size:%d\n", sum_rmse, line_process);
    return sqrt(sum_rmse / line_process);
}

float Predictor::validation(int step_length) {
    printf("start validation\n");

    float sum_rmse = 0;
    srand( time(NULL) );
    int line_idx = 0;
    int line_process = 0;
    int size_tot = (int) train_log.size();

    while (true) {
        line_idx += rand() % step_length + 1;
        if (line_idx >= size_tot) break;

        Line line = train_log[line_idx];
        float score_predict = predict(line.user_idx, line.item_idx);
        float err = line.score - score_predict;


        // !!! modify !!! for MAP
        sum_rmse += err * err / user_activity[line.user_idx];

        ++ line_process;
    }

    printf("after validation, sum_rmse:%f, size:%d\n", sum_rmse, line_process);
    return sqrt(sum_rmse / line_process);
}

