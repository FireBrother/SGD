//
// Created by 吴先 on 16/5/27.
//

#ifndef LR_H
#define LR_H


#include "SGD.h"
#include "utils.h"

class LR: public SGD{
public:
    LR();
private:
    float _p(const feature_t& X);
    float _cost(const feature_t& X, float y);
    float _tot_cost();
    weight_t _derived(const feature_t& X, float y);
};

LR::LR() {
    MAX_EPOCH = 50;
    THRESH_CONVERGE = 0.00001;
    ALPHA = 0.1;
}

float LR::_p(const feature_t &X) {
    float z = bias + dot_product(weight, X);
    float p = (float) sigmoid(z);
    return p;
}

float LR::_cost(const feature_t &X, float y) {
    return (float) -(y * log(_p(X) + 0.0001) + (1 - y) * log(1 - _p(X) + 0.0001));
}

float LR::_tot_cost() {
    float tot_cost = 0;
    for (size_t i = 0; i < X_s.size(); i++) tot_cost += _cost(X_s[i], y_s[i]);
    tot_cost /= X_s.size();
    return tot_cost;
}

weight_t LR::_derived(const feature_t& X, float y) {
    weight_t diff;
    double d = y - _p(X);
    for (auto p : X) {
        diff[p.first] += d * p.second;
    }
    return diff;
}



#endif //LR_H
