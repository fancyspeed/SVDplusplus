#include "Predictor.h"
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
using namespace std;

void Predictor::excute_testing(const char* path_test, const char* path_output) {
    printf("start to test\n");

    std::ifstream file_test(path_test);
    std::ofstream file_out(path_output);

    int line_process = 0;
    std::string line;

    while (getline(file_test, line)) {
        if ((++line_process) % 100000 == 0)
            printf("processed %d lines\n", line_process);

        int start_idx = 0;
        int end_idx = line.find(",");

        std::string userid_s = line.substr(start_idx, end_idx-start_idx);
        int cur_user = userid_2_idx[userid_s];

        multimap<float, string> sorted_items;

        while ((start_idx = end_idx + 1) < (int)line.length()) {
            end_idx = line.find(" ", start_idx);
            if (end_idx < 0)
                end_idx = (int)line.length();

            string itemid_s = line.substr(start_idx, end_idx-start_idx);
            int cur_item = itemid_2_idx[itemid_s];

            float score_predict = predict(cur_user, cur_item);
            sorted_items.insert(make_pair(score_predict, itemid_s));
        }

        multimap<float, string>::reverse_iterator it;
        int i_limit = 0;
        for (it = sorted_items.rbegin(); (i_limit) < 3 && it != sorted_items.rend(); ++it) {
            if (i_limit++ > 0) 
                file_out << " ";
            file_out << it->second; 
        }
        file_out << endl;
    }// end while

    file_out.close();
    file_test.close();
}

