// src/convolution.cpp
#include "convolution.h"
#include <vector>
#include <iostream>
#include <cmath> 

Convolution::Convolution(int input_channels, int output_channels, int kernel_size, int stride, int padding)
    : input_channels_(input_channels), output_channels_(output_channels), kernel_size_(kernel_size),
      stride_(stride), padding_(padding) {}

void Convolution::execute(const std::vector<std::vector<float>>& input_col,
                          const std::vector<std::vector<float>>& filters,
                          std::vector<std::vector<float>>& output,
                          const std::vector<float>& biases) {
    int num_filters = filters.size();
    int num_windows = input_col[0].size();

    // 初始化输出矩阵尺寸
    output.resize(num_filters, std::vector<float>(num_windows, 0));

    // 执行矩阵乘法
    for (int f = 0; f < num_filters; ++f) {
        for (int w = 0; w < num_windows; ++w) {
            float sum = 0.0f;
            for (int k = 0; k < filters[f].size(); ++k) {
                sum += filters[f][k] * input_col[k][w];
            }
            sum += biases[f];
            output[f][w] = sum;
        }
    }

    // 重构输出特征图
    int output_height = std::sqrt(num_windows);
    int output_width = output_height;  // 假设输出为方形
    for (int f = 0; f < num_filters; ++f) {
        std::cout << "Feature Map " << f << ":\n";
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                std::cout << output[f][y * output_width + x] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}
