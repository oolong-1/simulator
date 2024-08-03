#include "../npu/cim.h"
#include <vector>
#include <iostream>
#include <cmath>

void executeConvolution(const std::vector<std::vector<float>>& input_col,
                        const std::vector<std::vector<float>>& filters,
                        std::vector<std::vector<std::vector<float>>>& output,
                        const std::vector<float>& biases) {
    int num_filters = filters.size();  // 过滤器的数量
    int num_windows = input_col[0].size();  // 输入列的窗口数量

    // 初始化输出矩阵尺寸
    std::vector<std::vector<float>> temp_output(num_filters, std::vector<float>(num_windows, 0));

    // 执行矩阵乘法
    for (int f = 0; f < num_filters; ++f) {
        for (int w = 0; w < num_windows; ++w) {
            float sum = 0.0f;
            for (int k = 0; k < filters[f].size(); ++k) {
                sum += filters[f][k] * input_col[k][w];
            }
            sum += biases[f];  // 加上偏置
            temp_output[f][w] = sum;
        }
    }

    // 重构输出特征图
    int output_height = std::sqrt(num_windows);
    int output_width = output_height;  

    // 初始化重构后的输出
    output.resize(num_filters, std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0)));

    for (int f = 0; f < num_filters; ++f) {
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                output[f][y][x] = temp_output[f][y * output_width + x];
            }
        }
    }
}
