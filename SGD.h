//
// Created by 吴先 on 16/5/27.
//

#ifndef SGD_H
#define SGD_H


#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include "limonp/Logging.hpp"
#include "limonp/StringUtil.hpp"
#include "time.h"
#include "utils.h"

typedef std::unordered_map<size_t, float> feature_t;

class SGD {
public:
    void load_data(std::string filename);
    SGD() {};
    void train();

protected:
    std::vector<float> y_s;
    std::vector<feature_t> X_s;
    std::unordered_map<size_t, float> weight;
    float bias;

    size_t MAX_EPOCH = 50;
    float THRESH_CONVERGE = 10;
    float ALPHA = 1;

    void _init_weight();

    virtual float _p(const feature_t& X) = 0;
    virtual float _cost(const feature_t& X, float y) = 0;
    virtual float _tot_cost() = 0;
};

void SGD::load_data(std::string filename) {
    XLOG(INFO) << "Load data from " << filename;
    const time_t st = time(NULL);
    X_s.clear();
    y_s.clear();
    std::ifstream fin(filename);
    if (!fin) {
        XLOG(FATAL) << "Cannot open file " << filename;
    }
    std::string buff;
    int n = 0;
    while (std::getline(fin, buff)) {
        n++;
        std::vector<std::string> temp;
        limonp::Split(buff, temp, " ");
        XCHECK(temp.size() > 0);
        float y = std::stof(temp[0]);
        feature_t X;
        for (size_t i = 1; i < temp.size(); i++) {
            std::vector<std::string> p;
            limonp::Split(temp[i], p, ":");
            XCHECK(p.size() == 2);
            size_t k = std::stoul(p[0]);
            float v = std::stof(p[1]);
            X[k] = v;
        }
        y_s.push_back(y);
        X_s.push_back(X);
    }
    const time_t et = time(NULL);
    XLOG(INFO) << string_format("Load data finished, %d records loaded: %ds.", n, et-st);
}

void SGD::_init_weight() {
    for (auto X : X_s)
        for (auto p : X)
            weight[p.first] = 0;
}

void SGD::train() {
    XLOG(INFO) << string_format("Train: MAX_EPOCH=%d, THRESH_CONVERGE=%f, ALPHA=%f",
                                MAX_EPOCH, THRESH_CONVERGE, ALPHA);
    const time_t st = time(NULL);
    float alpha = ALPHA;
    size_t epoch = 0;
    float cvg;
    float tot_cost = _tot_cost();
    _init_weight();
    while (true) {
        epoch++;
        XLOG(INFO) << string_format("epoch: %d, cost=%f, alpha=%f", epoch, tot_cost, alpha);


        float temp_cost = tot_cost;
        tot_cost = _tot_cost();
        cvg = (float) fabs(tot_cost - temp_cost);
        if (cvg < THRESH_CONVERGE) {
            XLOG(INFO) << "The cost function has converged.";
            break;
        }
        if (epoch >= MAX_EPOCH) {
            XLOG(INFO) << "Max epoch reached.";
            break;
        }
    }
    const time_t et = time(NULL);
    XLOG(INFO) << string_format("Training finished: %ds.", et-st);
}

#endif //SGD_H
