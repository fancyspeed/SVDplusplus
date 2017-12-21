#include "SVDBiasPredictor.h"
#include "SGD.h"

int main(int argc, char** argv) {
    const char* path_items = "../../data/pig/item.txt";
    const char* path_users = "../../data/pig/user_profile.txt";
    const char* path_train = "../../data/pig_mid/session_train.8.txt";
    const char* path_sns = "../../data/pig_mid/sns.3.txt";

    const char* path_candidate = "../../data/pig_mid/pig_candidate.txt";
    const char* path_out = "sample1.txt";
    if (argc >= 2) {
        path_out = argv[1];
    }

    printf("start SVDBiasPredictor\n");

    int dimension = 50;
    float u_0 = 0.05;
    float power_factor = 0.5;

    SVDBiasPredictor predictor(dimension);

    if (false == predictor.load_items(path_items))
        return 1;
    if (false == predictor.load_users(path_users))
        return 1;
    if (false == predictor.load_train(path_train))
        return 1;
    if (false == predictor.load_sns(path_sns))
        return 1;

    if (false == predictor.init_bubi())
        return 1;
    if (false == predictor.init_UV(u_0))
        return 1;
    if (false == predictor.init_user_item_line(power_factor))
        return 1;

    int max_iter = 60;
    int step_length = 2;
    float alpha = 0.002;
    float lambda = 0.004;
    float eplison = -0.0005;

    SGD sgd;
    sgd.gradientDescent(predictor, max_iter, step_length, alpha, lambda, eplison);

    predictor.excute_testing(path_candidate, path_out);

    return 0;
}


