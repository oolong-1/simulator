// src/preprocess.cpp
#include "preprocess.h"
#include <iostream>
#include <vector>

Preprocess::Preprocess(int pad_value, int kernel_size, int stride)
    : pad_value_(pad_value), kernel_size_(kernel_size), stride_(stride) {}

void Preprocess::img2col(const std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<float>>& col_output) {
    int channels = input.size();
    int height = input[0].size();
    int width = input[0][0].size();
    int output_height = (height - kernel_size_) / stride_ + 1;
    int output_width = (width - kernel_size_) / stride_ + 1;

    int col_vector_size = channels * kernel_size_ * kernel_size_;
    col_output.resize(col_vector_size, std::vector<float>(output_height * output_width, 0));

    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            int col_index = y * output_width + x;
            int col_count = 0;
            for (int c = 0; c < channels; ++c) {
                for (int ky = 0; ky < kernel_size_; ++ky) {
                    for (int kx = 0; kx < kernel_size_; ++kx) {
                        int in_y = y * stride_ + ky;
                        int in_x = x * stride_ + kx;
                        col_output[col_count++][col_index] = input[c][in_y][in_x];
                    }
                }
            }
        }
    }
}
