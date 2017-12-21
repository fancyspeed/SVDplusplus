#include "SVDBiasPredictor.h"
#include <stdio.h>
#include <math.h>
using namespace std;

bool SVDBiasPredictor::init_bubi() {
    printf("start to initialize Bu Bi\n");

    float * NU = new float[num_users];
    float * NI = new float[num_items];
    BU = new float[num_users];
    BI = new float[num_items];
    if (!BU || !BI) {
        fprintf(stderr, "get memory error!\n");
        return false;
    }

    for (int i=0; i < num_users; ++i) 
        BU[i] = 0;
    for (int i=0; i < num_items; ++i)
        BI[i] = 0;
    for (int i=0; i < num_users; ++i) 
        NU[i] = 0;
    for (int i=0; i < num_items; ++i)
        NI[i] = 0;

    //init mu, b(u), b(i)
    mu = 0;
    for (int i=0; i<(int)train_log.size(); ++i) 
        mu += train_log[i].score;
    mu /= (int)train_log.size();

    for (int i=0; i<(int)train_log.size(); ++i) {
        int idx = train_log[i].user_idx;
        BU[idx] += train_log[i].score - mu;
        NU[idx] ++;
    }
    for (int i=0; i<num_users; ++i) {
        if (NU[i] > 0)
            BU[i] /= NU[i];
    }
    for (int i=0; i<(int)train_log.size(); ++i) {
        int idx = train_log[i].item_idx;
        BI[idx] += train_log[i].score - mu - BU[train_log[i].user_idx];
        NI[idx] ++;
    }
    for (int i=0; i<num_items; ++i) {
        if (NI[i] > 0)
            BI[i] /= NI[i];
    }

    delete []NI;
    delete []NU;

    printf("init Bu Bi finished, mu:%f\n", mu);
    return true;
}

bool SVDBiasPredictor::init_UV(float u_0) {
    printf("start to initialize UV, u_0:%f\n", u_0);

    //init U, V
    int n_p = num_users * dimension;
    int n_q = num_items * dimension;

    srand( time(NULL) );

    U = new float[n_p];
    V = new float[n_q];
    if (!U || !V) {
        fprintf(stderr, "get memory error!\n");
        return false;
    }

    for (int i=0; i < n_p; ++i) 
        U[i] = (rand() % 10000) / 10000.0 * u_0;
    for (int i=0; i < n_q; ++i)
        V[i] = (rand() % 10000) / 10000.0 * u_0;

    //init Y
    Y = new float[n_q];
    if (!Y) {
        fprintf(stderr, "get memory error!\n");
        return false;
    }

    for (int i=0; i < n_q; ++i)
        Y[i] = (rand() % 10000) / 10000.0 * u_0;

    //init V_sns, Y_sns
    V_sns = new float[n_q];
    if (!V_sns) {
        fprintf(stderr, "get memory error!\n");
        return false;
    }
    for (int i=0; i < n_q; ++i)
        V_sns[i] = (rand() % 10000) / 10000.0 * u_0;

    Y_sns = new float[n_q];
    if (!Y_sns) {
        fprintf(stderr, "get memory error!\n");
        return false;
    }
    for (int i=0; i < n_q; ++i)
        Y_sns[i] = (rand() % 10000) / 10000.0 * u_0;

    return true;
}


bool SVDBiasPredictor::init_user_item_line(float factor) {
    printf("start to initialize user->items\n");
    
    //init user->items
    for (int i=0; i<num_users; ++i) {
        map<int, int> items;
        user_item_line.insert(make_pair(i, items));
    }
    for (int i=0; i<(int)train_log.size(); ++i) {
        Line line = train_log[i];
        if (line.score > 0)
            user_item_line[line.user_idx].insert(make_pair(line.item_idx, i));
    }

    int n_user_sns = 0;
    for (int i=0; i<num_users; ++i) {
        if ((int) user_item_line[i].size() > 0)
            ++ n_user_sns;
    }
    printf("after train log, %d users have items\n", n_user_sns);

    //init Item popularity
    for (int i=0; i<num_items; ++i)
        item_popular.insert(make_pair(i, 0));
    for (int i=0; i<(int) train_log.size(); ++i)
        if (train_log[i].score > 0)
            item_popular[train_log[i].item_idx] ++;
    for (int i=0; i<num_items; ++i)
        item_popular[i] = pow(item_popular[i] + 1.0, factor);

    //init user activity
    for (int i=0; i<num_users; ++i)
        user_activity.insert(make_pair(i, 0));
    for (int i=0; i<(int) train_log.size(); ++i)
        if (train_log[i].score > 0)
            user_activity[train_log[i].user_idx] ++;
    for (int i=0; i<num_users; ++i)
        user_activity[i] = pow(user_activity[i] + 1.0, factor);

    //init user->items
    for (int i=0; i<num_users; ++i) {
        map<int, int> items;
        user_item_line_sns.insert(make_pair(i, items));
    }
    for (int i=0; i<(int)sns_log.size(); ++i) {
        Line line = sns_log[i];
        if (line.score > 0)
            user_item_line_sns[line.user_idx].insert(make_pair(line.item_idx, i));
    }

    n_user_sns = 0;
    for (int i=0; i<num_users; ++i) {
        if ((int) user_item_line_sns[i].size() > 0)
            ++ n_user_sns;
    }
    printf("after sns log, %d users have items\n", n_user_sns);

    //init Item popularity
    for (int i=0; i<num_items; ++i)
        item_popular_sns.insert(make_pair(i, 0));
    for (int i=0; i<(int) sns_log.size(); ++i)
        if (sns_log[i].score > 0)
            item_popular_sns[sns_log[i].item_idx] ++;
    for (int i=0; i<num_items; ++i)
        item_popular_sns[i] = pow(item_popular_sns[i] + 1.0, factor);

    //init user activity
    for (int i=0; i<num_users; ++i)
        user_activity_sns.insert(make_pair(i, 0));
    for (int i=0; i<(int) sns_log.size(); ++i)
        if (sns_log[i].score > 0)
            user_activity_sns[sns_log[i].user_idx] ++;
    for (int i=0; i<num_users; ++i)
        user_activity_sns[i] = pow(user_activity_sns[i] + 1.0, factor);

    return true;
}

