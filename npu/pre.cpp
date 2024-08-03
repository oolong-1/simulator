#include "../npu/pre.h"
#include <iostream>
#include <vector>

// img2col 函数实现：将输入的三维向量转换为列向量
void img2col(const std::vector<std::vector<std::vector<float>>>& img_input, 
             std::vector<std::vector<float>>& col_output, 
             int kernel_size, int kernel_stride) {
    int channels = img_input.size();
    int height = img_input[0].size();
    int width = img_input[0][0].size();
    int output_height = (height - kernel_size) / kernel_stride + 1;
    int output_width = (width - kernel_size) / kernel_stride + 1;

    int col_vector_size = channels * kernel_size * kernel_size;
    col_output.resize(col_vector_size, std::vector<float>(output_height * output_width, 0));

    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            int col_index = y * output_width + x;
            int col_count = 0;
            for (int c = 0; c < channels; ++c) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int in_y = y * kernel_stride + ky;
                        int in_x = x * kernel_stride + kx;
                        col_output[col_count++][col_index] = img_input[c][in_y][in_x];
                    }
                }
            }
        }
    }
}

// flattenFilters 函数实现：将四维过滤器向量展平成二维向量
void flattenFilters(const std::vector<std::vector<std::vector<std::vector<float>>>>& filters,
                    std::vector<std::vector<float>>& flattened_filters) {
    int num_filters = filters.size();
    int depth = filters[0].size();
    int height = filters[0][0].size();
    int width = filters[0][0][0].size();

    flattened_filters.resize(num_filters, std::vector<float>(depth * height * width, 0));

    for (int f = 0; f < num_filters; ++f) {
        int index = 0;
        for (int d = 0; d < depth; ++d) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    flattened_filters[f][index++] = filters[f][d][h][w];
                }
            }
        }
    }
}

// padControl 函数实现：对输入的三维向量进行填充
void padControl(const std::vector<std::vector<std::vector<float>>>& pad_input,
                std::vector<std::vector<std::vector<float>>>& pad_output,
                int pad_value) {
    int channels = pad_input.size();
    int height = pad_input[0].size();
    int width = pad_input[0][0].size();
    int padded_height = height + 2 * pad_value;
    int padded_width = width + 2 * pad_value;

    pad_output.resize(channels, std::vector<std::vector<float>>(padded_height, std::vector<float>(padded_width, 0)));

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                pad_output[c][h + pad_value][w + pad_value] = pad_input[c][h][w];
            }
        }
    }
}
