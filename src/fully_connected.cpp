// src/fully_connected.cpp
#include "../include/fully_connected.h"
#include <vector>

// 默认构造函数
FullyConnected::FullyConnected() {}

// 设置全连接层参数
void FullyConnected::setParams(const std::vector<std::vector<float>>& weights, const std::vector<float>& biases) {
    weights_ = weights;
    biases_ = biases;
}

// 执行全连接层
void FullyConnected::execute(const std::vector<float>& input, std::vector<float>& output) const {
    int num_neurons = biases_.size();
    output.resize(num_neurons, 0.0f);

    for (int i = 0; i < num_neurons; ++i) {
        for (int j = 0; j < input.size(); ++j) {
            output[i] += input[j] * weights_[i][j];
        }
        output[i] += biases_[i];
    }
}
