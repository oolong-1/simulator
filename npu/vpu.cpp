#include "../npu/vpu.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

// 添加一些打印语句来帮助调试
#pragma message("Compiling vpu.cpp")

// 激活函数实现
void applyActivation(std::vector<std::vector<std::vector<float>>>& input, const std::string& type) {
    for (auto& channel : input) {
        for (auto& row : channel) {
            for (auto& val : row) {
                if (type == "RELU") {
                    val = std::max(0.0f, val);  // ReLU激活
                } else if (type == "LEAKY_RELU") {
                    val = std::max(0.01f * val, val);  // Leaky ReLU激活
                } else if (type == "SIGMOID") {
                    val = 1.0f / (1.0f + std::exp(-val));  // Sigmoid激活
                } else if (type == "TANH") {
                    val = std::tanh(val);  // Tanh激活
                } else {
                    // NONE or invalid type: do nothing
                }
            }
        }
    }
}

// 最大池化实现
void maxPooling(const std::vector<std::vector<std::vector<float>>>& input,
                std::vector<std::vector<std::vector<float>>>& output,
                int pool_size, int pool_stride) {
    int in_channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int output_height = (input_height - pool_size) / pool_stride + 1;
    int output_width = (input_width - pool_size) / pool_stride + 1;

    output.resize(in_channels, std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0)));

    for (int c = 0; c < in_channels; ++c) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                float max_val = std::numeric_limits<float>::lowest();
                for (int kh = 0; kh < pool_size; ++kh) {
                    for (int kw = 0; kw < pool_size; ++kw) {
                        int ih = h * pool_stride + kh;
                        int iw = w * pool_stride + kw;
                        max_val = std::max(max_val, input[c][ih][iw]);  // 找到最大值
                    }
                }
                output[c][h][w] = max_val;  // 存储最大值
            }
        }
    }
}

// 平均池化实现
void averagePooling(const std::vector<std::vector<std::vector<float>>>& input,
                    std::vector<std::vector<std::vector<float>>>& output,
                    int pool_size, int pool_stride) {
    int in_channels = input.size();
    int input_height = input[0].size();
    // int input_width = input[0][0].size();
    int output_height = (input_height - pool_size) / pool_stride + 1;
    int output_width = output_height; // Assuming output is square

    output.resize(in_channels, std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0)));

    for (int c = 0; c < in_channels; ++c) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                float sum_val = 0.0f;
                for (int kh = 0; kh < pool_size; ++kh) {
                    for (int kw = 0; kw < pool_size; ++kw) {
                        int ih = h * pool_stride + kh;
                        int iw = w * pool_stride + kw;
                        sum_val += input[c][ih][iw];  // 累加池化窗口内的值
                    }
                }
                output[c][h][w] = sum_val / (pool_size * pool_size);  // 计算平均值
            }
        }
    }
}

// 池化函数实现
void executePooling(const std::vector<std::vector<std::vector<float>>>& input,
                    std::vector<std::vector<std::vector<float>>>& output,
                    const std::string& pool_type, int pool_size, int pool_stride) {
    if (pool_type == "MAX") {
        maxPooling(input, output, pool_size, pool_stride);  // 最大池化
    } else if (pool_type == "AVERAGE") {
        averagePooling(input, output, pool_size, pool_stride);  // 平均池化
    } else {
        std::cerr << "Unsupported pooling type: " << pool_type << std::endl;
    }
}
