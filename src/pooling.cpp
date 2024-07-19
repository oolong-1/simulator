// src/pooling.cpp
#include "../include/pooling.h"
#include <algorithm>
#include <limits>

// 默认构造函数
Pooling::Pooling() {}

// 设置池化参数
void Pooling::setParams(const PoolingParams& params) {
    params_ = params;
}

// 执行池化操作
void Pooling::execute(const std::vector<std::vector<std::vector<float>>>& input,
                      std::vector<std::vector<std::vector<float>>>& output) const {
    switch (params_.type) {
        case PoolingType::MAX:
            maxPooling(input, output);
            break;
        case PoolingType::AVERAGE:
            averagePooling(input, output);
            break;
    }
}

// 最大池化实现
void Pooling::maxPooling(const std::vector<std::vector<std::vector<float>>>& input,
                         std::vector<std::vector<std::vector<float>>>& output) const {
    int in_channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int output_height = (input_height - params_.pool_size) / params_.pool_stride + 1;
    int output_width = (input_width - params_.pool_size) / params_.pool_stride + 1;

    output.resize(in_channels, std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0)));

    for (int c = 0; c < in_channels; ++c) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                float max_val = std::numeric_limits<float>::lowest();
                for (int kh = 0; kh < params_.pool_size; ++kh) {
                    for (int kw = 0; kw < params_.pool_size; ++kw) {
                        int ih = h * params_.pool_stride + kh;
                        int iw = w * params_.pool_stride + kw;
                        max_val = std::max(max_val, input[c][ih][iw]);
                    }
                }
                output[c][h][w] = max_val;
            }
        }
    }
}

// 平均池化实现
void Pooling::averagePooling(const std::vector<std::vector<std::vector<float>>>& input,
                             std::vector<std::vector<std::vector<float>>>& output) const {
    int in_channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int output_height = (input_height - params_.pool_size) / params_.pool_stride + 1;
    int output_width = (input_width - params_.pool_size) / params_.pool_stride + 1;

    output.resize(in_channels, std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0)));

    for (int c = 0; c < in_channels; ++c) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                float sum_val = 0.0f;
                for (int kh = 0; kh < params_.pool_size; ++kh) {
                    for (int kw = 0; kw < params_.pool_size; ++kw) {
                        int ih = h * params_.pool_stride + kh;
                        int iw = w * params_.pool_stride + kw;
                        sum_val += input[c][ih][iw];
                    }
                }
                output[c][h][w] = sum_val / (params_.pool_size * params_.pool_size);
            }
        }
    }
}
