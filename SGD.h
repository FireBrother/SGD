//
// Created by 吴先 on 16/5/27.
// Last Updated on 17/11/30.
// SGD基类，实现了基本的数据读入、训练和预测，
// 需要继承该基类并实现计算概率、loss和梯度的接口
//

#ifndef SGD_H
#define SGD_H


#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <ctime>
#include <random>
#include "limonp/Logging.hpp"
#include "limonp/StringUtil.hpp"
#include "utils.h"

// 为了实现方便，使用map作为特征的类型，相同的key表示在相同维度
// 实际上可以修改成vector，但是在O2的优化下，速度没什么差别
typedef std::unordered_map<size_t, float> feature_t;
typedef std::unordered_map<size_t, float> weight_t;

class SGD {
public:
    void load_train_data(const std::string &img_filename, const std::string &label_filename);
    void load_test_data(const std::string &img_filename, const std::string &label_filename);
    SGD() {};
    void train();
    float test();
    float predict(const feature_t &X);

protected:
    // 保存训练数据和测试数据
    std::vector<float> y_s;
    std::vector<feature_t> X_s;
    std::vector<float> y_test_s;
    std::vector<feature_t> X_test_s;
    // 参数向量，以及偏移量
    weight_t weight;
    float weight0;

    // 最大轮数、收敛阈值和学习率
    size_t MAX_EPOCH = 10;
    float THRESH_CONVERGE = 0.001;
    float ALPHA = 0.1;

    // 初始化权重和数据读取
    void _init_weight();
    std::tuple<std::vector<feature_t>, std::vector<float> > _load_data(const std::string &img_filename,
                                                                       const std::string &label_filename);

    virtual float _p(const feature_t& X) = 0;   // p(y|x)的值
    virtual float _cost(const feature_t& X, float y) = 0; // 给定一对样本，计算loss
    virtual float _tot_cost() = 0;
    virtual weight_t _derived(const feature_t& X, float y) = 0; // 给定一对样本，计算梯度

};

void SGD::load_train_data(const std::string &img_filename, const std::string &label_filename) {
    XLOG(INFO) << "Load train data from " << img_filename;
    const time_t st = time(NULL);
    std::tie(X_s, y_s) = _load_data(img_filename, label_filename);
    const time_t et = time(NULL);
    XLOG(INFO) << string_format("Load train data finished, %d records loaded: %ds.", X_s.size(), et-st);
}

void SGD::load_test_data(const std::string &img_filename, const std::string &label_filename) {
    XLOG(INFO) << "Load test data from " << img_filename;
    const time_t st = time(NULL);
    std::tie(X_test_s, y_test_s) = _load_data(img_filename, label_filename);
    const time_t et = time(NULL);
    XLOG(INFO) << string_format("Load test data finished, %d records loaded: %ds.", X_test_s.size(), et-st);
}

// 读入数据集，并返回一个特征向量矩阵与标签列表的tuple
// 这个_load_data是针对mnist数据集编写的，图片大小28*28像素，0～255灰阶，预先做了居中和旋转摆正，可以不从图像处理的角度考虑优化。
// 具体的数据描述可以参考http://yann.lecun.com/exdb/mnist/
// 返回值为X_s和y_s，
// X_s按行存储每张图片，每张图片以行优先的方式存储，每一维代表一个像素的灰度值（0～255），
// y_s为标签列表，取值范围0～9。
// 注意！这里只做了数字3的识别，即3为标签1，其他为标签0，作业要求实现对所有数字的识别，即输入一个手写数字图片，返回其最有可能的数字。
std::tuple<std::vector<feature_t>, std::vector<float> > SGD::_load_data(const std::string &img_filename,
                                                                                      const std::string &label_filename) {
    std::vector<float> y_s;
    std::vector<feature_t> X_s;
    std::ifstream img_fin(img_filename, std::ios::binary|std::ios::in);
    std::ifstream label_fin(label_filename, std::ios::binary|std::ios::in);
    if (!img_fin) {
        XLOG(FATAL) << "Cannot open file " << img_filename;
    }
    if (!label_fin) {
        XLOG(FATAL) << "Cannot open file " << label_filename;
    }
    int magic;
    int num_items;
    int num_rows, num_cols;
    // 因为是二进制存储的，需要考虑大小端的问题
    // 如果你的机器在大小端方面出现问题，可以考虑删除reverse。
    img_fin.read((char*)&magic, sizeof(magic));
    std::reverse((char*)&magic, (char*)&magic+sizeof(magic));
    img_fin.read((char*)&num_items, sizeof(num_items));
    std::reverse((char*)&num_items, (char*)&num_items+sizeof(num_items));
    img_fin.read((char*)&num_rows, sizeof(num_rows));
    std::reverse((char*)&num_rows, (char*)&num_rows+sizeof(num_rows));
    img_fin.read((char*)&num_cols, sizeof(num_cols));
    std::reverse((char*)&num_cols, (char*)&num_cols+sizeof(num_cols));

    label_fin.read((char*)&magic, sizeof(magic));
    std::reverse((char*)&magic, (char*)&magic+sizeof(magic));
    label_fin.read((char*)&num_items, sizeof(num_items));
    std::reverse((char*)&num_items, (char*)&num_items+sizeof(num_items));
    for (int i = 0; i < num_items; i++) {
        unsigned char buff;
        feature_t x;
        for (int j = 0; j < num_rows; j++) {
            for (int k = 0; k < num_cols; k++) {
                img_fin.read((char*)&buff, sizeof(buff));
                x[j*num_cols+k] = float(buff);
            }
        }
        label_fin.read((char*)&buff, sizeof(buff));
        X_s.push_back(x);
        // TODO:！！！注意！！！，这里就是只识别数字3的罪魁祸首
        y_s.push_back(float(buff) == 3 ? 1 : 0);
    }
    return std::make_tuple(X_s, y_s);
}

// 权重的随机初始化，范围-0.5~0.5
void SGD::_init_weight() {
    srand((unsigned int) time(NULL));
    weight0 = 0;
    for (auto X : X_s)
        for (auto p : X) {
            weight[p.first] = (float) (1.0 * (rand() % 1000) / 1000 - 0.5) ;
        }
}

void SGD::train() {
    XLOG(INFO) << string_format("Train: MAX_EPOCH=%d, THRESH_CONVERGE=%f, ALPHA=%f",
                                MAX_EPOCH, THRESH_CONVERGE, ALPHA);
    const time_t st = time(NULL);
    // 初始化参数向量和训练参数
    _init_weight();
    float alpha = ALPHA;    // 学习率
    size_t epoch = 0;       // 训练轮数
    float cvg;              // 两次训练的loos之差，用于提前停止
    float tot_cost = _tot_cost();   // 在整个训练集上的loss
    // 用于打乱序号，SGD每次训练开始时应该shuffle训练集
    std::vector<int> idx(X_s.size());
    for (int i = 0; i < X_s.size(); ++i) {
        idx.push_back (i);
    }
    while (true) {
        const time_t ep_st = time(NULL);
        epoch++;
        std::shuffle(idx.begin(), idx.end(), std::default_random_engine());
        // 每接受一对样本，执行一次梯度下降
        for (size_t i = 0; i < X_s.size(); i++) {
            weight_t diff = _derived(X_s[idx[i]], y_s[idx[i]]); // 计算梯度
            weight0 -= alpha * (_p(X_s[idx[i]]) - y_s[idx[i]]); // 根据梯度公式，gb=(p-sigmoid(w*x))*1
            for (auto p : diff) {
                weight[p.first] -= alpha * p.second;            // 根据梯度公式，gw=(p-sigmoid(w*x))*x
            }
        }
        // 计算当前模型的loss
        float temp_cost = tot_cost;
        tot_cost = _tot_cost();
        // 计算loss变化
        cvg = (float) fabs(tot_cost - temp_cost);
        const time_t ep_et = time(NULL);
        XLOG(INFO) << string_format("epoch: %d, cost=%f, alpha=%f: %ds", epoch, tot_cost, alpha, ep_et-ep_st);
        // 是否已收敛，如果收敛则提前停止
        if (cvg < THRESH_CONVERGE) {
            XLOG(INFO) << "The cost function has converged: cvg=" << cvg;
            break;
        }
        // 是否到达最大迭代轮数
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
