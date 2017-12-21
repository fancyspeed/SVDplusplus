#include "SVDBiasPredictor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
using namespace std;

bool SVDBiasPredictor::load_sns(const char* path_input) {
    cout << "start to load data: " << path_input << endl;

    ifstream file_input(path_input);
    if (!file_input) {
        fprintf(stderr, "open file failed\n");
        return false;
    }

    string str;
    int line_process = 0;

    while (getline(file_input, str)) {
        Line line;
        if (false == parse_rec_log(str, line))
            continue;

        sns_log.push_back(line);

        if ((++line_process) % 1000000 == 0)
            printf("processed %d lines\n", line_process);
    }

    file_input.close();
    cout << "load finished, get training log lines: " << (int)sns_log.size() << endl;
    return true;
}

