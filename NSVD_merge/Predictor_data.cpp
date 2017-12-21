#include "Predictor.h"
#include <iostream>
#include <fstream>
using namespace std;

void Predictor::parse_item_line(string& str) {
    static int itemid_idx = 0;
    int tab = str.find("\t", 0);
    string itemid_s = str.substr(0, tab);

    if (itemid_2_idx.end() == itemid_2_idx.find(itemid_s)) {
        itemid_2_idx[itemid_s] = itemid_idx ++;
    }
}

bool Predictor::load_items(const char* path_input) {
    cout << "start to load items: " << path_input << endl;

    ifstream file_input(path_input);
    if (!file_input) {
        fprintf(stderr, "open file failed\n");
        return false;
    }

    string str;
    int line_process = 0;

    while (getline(file_input, str)) {
        if ((++line_process) % 100000 == 0)
            printf("processed %d lines\n", line_process);
        parse_item_line(str);
    }

    num_items = (int) itemid_2_idx.size();
    printf("load items finished, size: %d\n", num_items); 
    return true;
}

void Predictor::parse_user_line(string& str) {
    static int userid_idx = 0;
    int tab = str.find("\t", 0);
    string userid_s = str.substr(0, tab);

    if (userid_2_idx.end() == userid_2_idx.find(userid_s)) {
        userid_2_idx[userid_s] = userid_idx ++;
    }
}

bool Predictor::load_users(const char* path_input) {
    cout << "start to load users: " << path_input << endl;

    ifstream file_input(path_input);
    if (!file_input) {
        fprintf(stderr, "open file failed\n");
        return false;
    }

    string str;
    int line_process = 0;

    while (getline(file_input, str)) {
        if ((++line_process) % 1000000 == 0)
            printf("processed %d lines\n", line_process);
        parse_user_line(str);
    }

    num_users = (int) userid_2_idx.size();
    printf("load users finished, size: %d\n", num_users); 
    return true;
}

bool Predictor::parse_rec_log(string& str, Line& line) {
    int tab0 = 0;
    int tab1 = str.find("\t", tab0+1);
    int tab2 = str.find("\t", tab1+1);
    int tab3 = str.find("\t", tab2+1);
    if (tab3 < 0) tab3 = (int) str.length();

    string userid_s = str.substr(tab0, tab1-tab0);
    if (userid_2_idx.end() == userid_2_idx.find(userid_s)) {
        return false;
    } 
    int cur_user = userid_2_idx[userid_s];

    string itemid_s = str.substr(tab1+1, tab2-tab1-1);
    if (itemid_2_idx.end() == itemid_2_idx.find(itemid_s)) {
        return false;
    } 
    int cur_item = itemid_2_idx[itemid_s];

    string score_s = str.substr(tab2+1, tab3-tab2-1);
    float score_i = atof(score_s.c_str());
    if (score_i < 0) score_i = 0;

    line.user_idx = cur_user;
    line.item_idx = cur_item;
    line.score = score_i;
    line.line_id = userid_s + "-" + itemid_s;

    return true;
}

bool Predictor::load_train(const char* path_input) {
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

        train_log.push_back(line);
        line_map.insert(make_pair(line.line_id, line.score));
        if (line.score > 0) n_positive ++;
        else n_negative ++;

        if ((++line_process) % 1000000 == 0)
            printf("processed %d lines, %d positive, %d negative\n", line_process, n_positive, n_negative);
    }

    file_input.close();
    cout << "load finished, get training log lines: " << (int)train_log.size() << endl;
    return true;
}

