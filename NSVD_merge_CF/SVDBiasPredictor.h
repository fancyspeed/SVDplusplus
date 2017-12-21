#pragma once
#include "Predictor.h"

class SVDBiasPredictor : public Predictor {
  protected:
    //mu and bias
    float mu;
    float * BU;
    float * BI;

    //U, V, Y
    int dimension;
    float * U;
    float * V;
    float * Y;

    //< useridx, <itemidx, lineidx> >
    std::map< int, std::map<int, int> > user_item_line;
    //user activity: (1 + np)^0.5
    std::map< int, float > user_activity;
    //item popularity: (1 + np)^gama
    std::map< int, float > item_popular;

    //sns data
    std::vector<Line> sns_log;
    //sns V, Y
    float * V_sns;
    float * Y_sns;
    //< useridx, <itemidx, lineidx> >
    std::map< int, std::map<int, int> > user_item_line_sns;
    //user activity: (1 + np)^0.5
    std::map< int, float > user_activity_sns;
    //item popularity: (1 + np)^gama
    std::map< int, float > item_popular_sns;

  public:
    SVDBiasPredictor(int dimension) 
        :  mu(0), BU(NULL), BI(NULL), 
           U(NULL), V(NULL), Y(NULL),
           V_sns(NULL), Y_sns(NULL) { 
            this->dimension = dimension;
    }
    virtual ~SVDBiasPredictor() {
        if (NULL != Y_sns)    delete []Y_sns;
        if (NULL != V_sns)    delete []V_sns;

        if (NULL != Y)    delete []Y;
        if (NULL != U)    delete []U;
        if (NULL != V)    delete []V;

        if (NULL != BU)    delete []BU;
        if (NULL != BI)    delete []BI;
    }

    bool load_sns(const char* path_input);

    bool init_bubi();
    bool init_UV(float u_0);
    bool init_user_item_line(float factor);

    float update(int line_idx, float alpha, float lambda);
    float predict(int user_idx, int item_idx);
};

