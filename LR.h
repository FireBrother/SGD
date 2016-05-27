//
// Created by 吴先 on 16/5/27.
//

#ifndef LR_H
#define LR_H


#include "SGD.h"
#include "utils.h"

class LR: public SGD{
public:

private:
    size_t MAX_EPOCH = 50;
    float THRESH_CONVERGE = 10;
    float ALPHA = 1;

    float _p(const feature_t& X);
    float _cost(const feature_t& X, float y);
    float _tot_cost();
};

float LR::_p(const feature_t &X) {
    float z = bias + dot_product(weight, X);
    return (float) sigmoid(z);
}

float LR::_cost(const feature_t &X, float y) {
    return (float) (-y * log(_p(X)) - (1 - y) * log(1 - _p(X)));
}

float LR::_tot_cost() {
    float tot_cost = 0;
    for (size_t i = 0; i < X_s.size(); i++) tot_cost += _cost(X_s[i], y_s[i]);
    tot_cost /= X_s.size();
    return tot_cost;
}




#endif //LR_H
