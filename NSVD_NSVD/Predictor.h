#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <vector>

struct Line {
  int user_idx;
  int item_idx;
  float score;
  int time;
  std::string line_id;
};

class Predictor {
  protected:
    //train data
    std::vector<Line> train_log;
    int num_users;
    int num_items;

    //user activity: (1 + np)^0.5
    std::map< int, float > user_activity;
    //item popularity: (1 + np)^gama
    std::map< int, float > item_popular;

    //userid/itemid mapping
    std::map<std::string, int> userid_2_idx;
    std::map<std::string, int> itemid_2_idx;
    int cur_user_idx;
    int cur_item_idx;

    std::map<std::string, float> line_map;
    int n_positive;
    int n_negative;

  public:
    Predictor() : num_users(0), num_items(0),
          cur_user_idx(0), cur_item_idx(0),
          n_positive(0), n_negative(0) { }
    virtual ~Predictor() { }

    void parse_item_line(std::string& str);
    virtual bool load_items(const char* path_input);
    void parse_user_line(std::string& str);
    virtual bool load_users(const char* path_input);
    bool parse_rec_log(std::string& str, Line& line);
    virtual bool load_train(const char* path_input);

    virtual float train(int step_length, float alpha, float lambda);
    virtual float validation(int step_length);

    virtual float update(int line_idx, float alpha, float lambda) = 0;
    virtual float predict(int user_idx, int item_idx) = 0;

    virtual void excute_testing(const char* path_test, const char* path_output);
};

