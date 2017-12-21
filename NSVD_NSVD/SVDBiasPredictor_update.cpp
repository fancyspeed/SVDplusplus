#include "SVDBiasPredictor.h"
#include <stdio.h>
#include <math.h>
using namespace std;

float SVDBiasPredictor::update(int line_idx, float alpha, float lambda) {
    Line line = train_log[line_idx];
    float score_predict = predict(line.user_idx, line.item_idx);
    float err = line.score - score_predict;

    // !!! modify !!! for MAP
    alpha /= user_activity[line.user_idx]; 

    int v_idx_0, y_idx_0, v_idx, y_idx;
    int eigen_i;
    v_idx_0 = line.item_idx * dimension;

    static float q_temp[1000];

    //update BU, BI
    //BU[line.user_idx] += alpha * ( err - lambda * BU[line.user_idx] );
    //BI[line.item_idx] += alpha * ( err - lambda * BI[line.item_idx] );
    
    //NSVD-train
    for (int eigen_i = 0; eigen_i < dimension; ++ eigen_i) {
        v_idx = v_idx_0 + eigen_i;
        q_temp[eigen_i] = V[v_idx];
        V[v_idx] -= alpha * lambda * V[v_idx];
    }

    map<int, int>& item_line = user_item_line[line.user_idx];
    map<int, int>::iterator it;
    float factor = err / user_activity[line.user_idx]; 
    if (user_activity[line.user_idx] < 0.5)
        factor = 0;

    for (it=item_line.begin(); it!=item_line.end(); ++it) {
        y_idx_0 = it->first * dimension;

        for (eigen_i = 0; eigen_i < dimension; ++ eigen_i) {
            v_idx = v_idx_0 + eigen_i;
            y_idx = y_idx_0 + eigen_i;
            V[v_idx] += alpha * factor * Y[y_idx]; 
            Y[y_idx] += alpha * factor * q_temp[eigen_i] - alpha * lambda * Y[y_idx];
        }
    }

    //NSVD-sns
    for (int eigen_i = 0; eigen_i < dimension; ++ eigen_i) {
        v_idx = v_idx_0 + eigen_i;
        q_temp[eigen_i] = V_sns[v_idx];
        V_sns[v_idx] -= alpha * lambda * V_sns[v_idx];
    }

    item_line = user_item_line_sns[line.user_idx];
    factor = err / user_activity_sns[line.user_idx]; 
    if (user_activity_sns[line.user_idx] > 0.5)
        factor = err / user_activity_sns[line.user_idx]; 
    else
        factor = 0;

    for (it=item_line.begin(); it!=item_line.end(); ++it) {
        y_idx_0 = it->first * dimension;

        for (eigen_i = 0; eigen_i < dimension; ++ eigen_i) {
            v_idx = v_idx_0 + eigen_i;
            y_idx = y_idx_0 + eigen_i;
            V_sns[v_idx] += alpha * factor * Y_sns[y_idx]; 
            Y_sns[y_idx] += alpha * factor * q_temp[eigen_i] - alpha * lambda * Y_sns[y_idx];
        }
    }

    return err * err;
}

float SVDBiasPredictor::predict(int user_idx, int item_idx) {
    //Bias
    float score_predict = mu;
    score_predict += BU[user_idx];
    score_predict += BI[item_idx];
    
    //NSVD-train
    float nsvd_score = 0;

    map<int, int>& item_line = user_item_line[user_idx];
    map<int, int>::iterator it;

    int v_idx_0, y_idx_0, v_idx, y_idx;
    int eigen_i;
    v_idx_0 = item_idx * dimension;

    for (it=item_line.begin(); it!=item_line.end(); ++it) {
        y_idx_0 = it->first * dimension;

        for (eigen_i = 0; eigen_i < dimension; ++ eigen_i) {
            v_idx = v_idx_0 + eigen_i;
            y_idx = y_idx_0 + eigen_i;
            nsvd_score += V[v_idx] * Y[y_idx];
        }
    }
    if (user_activity[user_idx] > 0.5)
        score_predict += nsvd_score / user_activity[user_idx];

    //NSVD-sns
    float nsvd_score_sns = 0;

    item_line = user_item_line_sns[user_idx];

    for (it=item_line.begin(); it!=item_line.end(); ++it) {
        y_idx_0 = it->first * dimension;

        for (eigen_i = 0; eigen_i < dimension; ++ eigen_i) {
            v_idx = v_idx_0 + eigen_i;
            y_idx = y_idx_0 + eigen_i;
            nsvd_score_sns += V_sns[v_idx] * Y_sns[y_idx];
        }
    }

    if (user_activity_sns[user_idx] > 0.5)
        score_predict += nsvd_score_sns / user_activity_sns[user_idx];

    return score_predict;
}

