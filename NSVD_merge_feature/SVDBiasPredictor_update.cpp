#include "SVDBiasPredictor.h"
#include <stdio.h>
#include <math.h>
using namespace std;

float SVDBiasPredictor::update(int line_idx, float alpha, float lambda) {
    Line line = train_log[line_idx];
    float score_predict = predict(line.user_idx, line.item_idx);
    float err = line.score - score_predict;

    int v_idx_0, y_idx_0, v_idx, y_idx;
    int eigen_i;
    v_idx_0 = line.item_idx * dimension;

    static float q_temp[1000];

    //update BU, BI
    BU[line.user_idx] += alpha * ( err - lambda * BU[line.user_idx] );
    BI[line.item_idx] += alpha * ( err - lambda * BI[line.item_idx] );
    int age = user_age[line.user_idx];
    int gender = user_gender[line.user_idx];
    B_age[age] += alpha * ( err - lambda * B_age[age] );
    B_gender[gender] += alpha * ( err - lambda * B_gender[gender] );
    
    //V
    for (int eigen_i = 0; eigen_i < dimension; ++ eigen_i) {
        v_idx = v_idx_0 + eigen_i;
        q_temp[eigen_i] = V[v_idx];
        V[v_idx] -= alpha * lambda * V[v_idx];
    }

    //Y: NSVD-train
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

    //Y: NSVD-sns
    item_line = user_item_line_sns[line.user_idx];
    if (user_activity_sns[line.user_idx] > 0.5)
        factor = err / user_activity_sns[line.user_idx]; 
    else
        factor = 0;

    for (it=item_line.begin(); it!=item_line.end(); ++it) {
        y_idx_0 = it->first * dimension;

        for (eigen_i = 0; eigen_i < dimension; ++ eigen_i) {
            v_idx = v_idx_0 + eigen_i;
            y_idx = y_idx_0 + eigen_i;
            V[v_idx] += alpha * factor * Y_sns[y_idx]; 
            Y_sns[y_idx] += alpha * factor * q_temp[eigen_i] - alpha * lambda * Y_sns[y_idx];
        }
    }
    return err * err;
}

float SVDBiasPredictor::predict(int user_idx, int item_idx) {
    //Bias
    float score_predict = mu + BU[user_idx] + BI[item_idx];
    int age = user_age[user_idx];
    int gender = user_gender[user_idx];
    score_predict += B_age[age] + B_gender[gender];
    
    int v_idx_0, y_idx_0, v_idx, y_idx;
    int eigen_i;
    v_idx_0 = item_idx * dimension;

    //NSVD-train
    float nsvd_score = 0;

    map<int, int>& item_line = user_item_line[user_idx];
    map<int, int>::iterator it;

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
            nsvd_score_sns += V[v_idx] * Y_sns[y_idx];
        }
    }
    if (user_activity_sns[user_idx] > 0.5)
        score_predict += nsvd_score_sns / user_activity_sns[user_idx];

    return score_predict;
}

