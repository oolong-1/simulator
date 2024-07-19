// src/batch_normalization.cpp
#include "../include/batch_normalization.h"
#include <algorithm>
#include <cmath>
#include <numeric>

// 默认构造函数
BatchNormalization::BatchNormalization() : epsilon_(1e-5) {}

// 设置批量归一化参数
void BatchNormalization::setParams(float epsilon, const std::vector<float>& gamma, const std::vector<float>& beta) {
    epsilon_ = epsilon;
    gamma_ = gamma;
    beta_ = beta;
}

// 执行批量归一化
void BatchNormalization::execute(const std::vector<std::vector<std::vector<float>>>& input,
                                 std::vector<std::vector<std::vector<float>>>& output) const {
    int channels = input.size();
    int height = input[0].size();
    int width = input[0][0].size();

    output.resize(channels, std::vector<std::vector<float>>(height, std::vector<float>(width, 0)));

    for (int c = 0; c < channels; ++c) {
        // 计算均值
        float mean = 0.0f;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                mean += input[c][h][w];
            }
        }
        mean /= (height * width);

        // 计算方差
        float variance = 0.0f;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                variance += (input[c][h][w] - mean) * (input[c][h][w] - mean);
            }
        }
        variance /= (height * width);

        // 归一化并缩放和平移
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                output[c][h][w] = gamma_[c] * (input[c][h][w] - mean) / std::sqrt(variance + epsilon_) + beta_[c];
            }
        }
    }
}
