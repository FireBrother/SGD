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
typedef std::unordered_map<size_t, float> weight_t;

class SGD {
public:
    void load_train_data(std::string filename);
    void load_test_data(std::string filename);
    SGD() {};
    void train();
    float test();
    float predict(const feature_t &X);

protected:
    std::vector<float> y_s;
    std::vector<feature_t> X_s;
    std::vector<float> y_test_s;
    std::vector<feature_t> X_test_s;
    weight_t weight;
    float weight0;

    size_t MAX_EPOCH = 10;
    float THRESH_CONVERGE = 0.001;
    float ALPHA = 0.1;
    float LAMBDA = 1;

    void _init_weight();
    std::tuple<std::vector<feature_t>, std::vector<float>, int, int, int> _load_data(std::string filename);

    virtual float _p(const feature_t& X) = 0;
    virtual float _cost(const feature_t& X, float y) = 0;
    virtual float _tot_cost() = 0;
    virtual weight_t _derived(const feature_t& X, float y) = 0;

};

void SGD::load_train_data(std::string filename) {
    XLOG(INFO) << "Load train data from " << filename;
    const time_t st = time(NULL);
    int n, npos, nneg;
    std::tie(X_s, y_s, n, npos, nneg) = _load_data(filename);
    const time_t et = time(NULL);
    XLOG(INFO) << string_format("Load train data finished, %d(%d pos, %d neg) records loaded: %ds.", n, npos, nneg, et-st);
}

void SGD::load_test_data(std::string filename) {
    XLOG(INFO) << "Load test data from " << filename;
    const time_t st = time(NULL);
    int n, npos, nneg;
    std::tie(X_test_s, y_test_s, n, npos, nneg) = _load_data(filename);
    const time_t et = time(NULL);
    XLOG(INFO) << string_format("Load test data finished, %d(%d pos, %d neg) records loaded: %ds.", n, npos, nneg, et-st);
}

std::tuple<std::vector<feature_t>, std::vector<float>, int, int, int> SGD::_load_data(std::string filename) {
    std::vector<float> y_s;
    std::vector<feature_t> X_s;
    std::ifstream fin(filename);
    if (!fin) {
        XLOG(FATAL) << "Cannot open file " << filename;
    }
    std::string buff;
    int n = 0, npos = 0, nneg = 0;
    while (std::getline(fin, buff)) {
        n++;
        std::vector<std::string> temp;
        limonp::Split(buff, temp, " ");
        XCHECK(temp.size() > 0);
        float y = std::stof(temp[0]);
        if (y == 1) npos++;
        else {
            y = 0;
            nneg++;
        }
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
    return make_tuple(X_s, y_s, n, npos, nneg);
}

void SGD::_init_weight() {
    srand((unsigned int) time(NULL));
    weight0 = 0;
    for (auto X : X_s)
        for (auto p : X) {
            weight[p.first] = (float) (1.0 * (rand() % 1000) / 1000000 - 0.0005) ;
        }
}

void SGD::train() {
    XLOG(INFO) << string_format("Train: MAX_EPOCH=%d, THRESH_CONVERGE=%f, ALPHA=%f",
                                MAX_EPOCH, THRESH_CONVERGE, ALPHA);
    const time_t st = time(NULL);
    _init_weight();
    float alpha = ALPHA;
    size_t epoch = 0;
    float cvg;
    float tot_cost = _tot_cost();
    while (true) {
        const time_t ep_st = time(NULL);
        epoch++;
        for (size_t i = 0; i < X_s.size(); i++) {
            weight_t diff = _derived(X_s[i], y_s[i]);
            weight0 -= alpha * (_p(X_s[i]) - y_s[i]);
            for (auto p : diff) {
                weight[p.first] -= alpha * p.second;
            }
        }
        float temp_cost = tot_cost;
        tot_cost = _tot_cost();
        cvg = (float) fabs(tot_cost - temp_cost);
        const time_t ep_et = time(NULL);
        XLOG(INFO) << string_format("epoch: %d, cost=%f, alpha=%f: %ds", epoch, tot_cost, alpha, ep_et-ep_st);
        if (cvg < THRESH_CONVERGE) {
            XLOG(INFO) << "The cost function has converged: cvg=" << cvg;
            break;
        }
        if (epoch >= MAX_EPOCH) {
            XLOG(INFO) << "Max epoch reached: epoch=" << epoch;
            break;
        }
    }
    const time_t et = time(NULL);
    XLOG(INFO) << string_format("weight0=%f", weight0);
    XLOG(INFO) << string_format("Training finished, cost=%f: %ds.", _tot_cost(), et-st);
}

float SGD::test() {
    XLOG(INFO) << string_format("Test: num=%d.", X_test_s.size());
    const time_t st = time(NULL);
    int c = 0;
    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < X_test_s.size(); i++) {
        float y_star = predict(X_test_s[i]);
        if (y_star == y_test_s[i]) c++;
        if (y_star == 1.0 && y_test_s[i] == 1.0) tp++;
        else if (y_star == 1.0 && y_test_s[i] == 0.0) fp++;
        else if (y_star == 0.0 && y_test_s[i] == 1.0) fn++;
        else if (y_star == 0.0 && y_test_s[i] == 0.0) tn++;
    }
    float acc = (float) (1.0 * c / X_test_s.size());
    float p, r, f;
    p = (float) (1.0 * tp / (tp + fp));
    r = (float) (1.0 * tp / (tp + fn));
    f = (float) (2.0 * p * r / (p + r));
    const time_t et = time(NULL);
    XLOG(INFO) << string_format("precision=%f%%, recall=%f%%, f-measure=%f", p * 100, r * 100, f);
    XLOG(INFO) << string_format("Testing finished, acc=%f%%: %ds.", acc * 100, et-st);
    return acc;
}

float SGD::predict(const feature_t& X) {
    float p = _p(X);
    //XLOG(INFO) << p;
    return (float) (p >= 0.5 ? 1.0 : 0.0);
}


#endif //SGD_H
