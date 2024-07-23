// src/preprocess.cpp
#include "../include/preprocess.h"
#include <vector>
#include <algorithm>

// 构造函数，初始化tile大小和填充值
Preprocess::Preprocess(int tile_size, int pad_value) : tile_size_(tile_size), pad_value_(pad_value) {}

// Tile读取控制
void Preprocess::tileReadControl(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& tiled_output) {
    int height = input.size();
    int width = input[0].size();
    int new_height = (height + tile_size_ - 1) / tile_size_;
    int new_width = (width + tile_size_ - 1) / tile_size_;
    
    tiled_output.resize(new_height * tile_size_, std::vector<float>(new_width * tile_size_, 0));
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            tiled_output[i][j] = input[i][j];
        }
    }
}

// Pad控制
void Preprocess::padControl(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& padded_output) {
    int height = input.size();
    int width = input[0].size();
    int padded_height = height + 2 * pad_value_;
    int padded_width = width + 2 * pad_value_;
    
    padded_output.resize(padded_height, std::vector<float>(padded_width, 0));
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            padded_output[i + pad_value_][j + pad_value_] = input[i][j];
        }
    }
}

// Line buffer读写控制
void Preprocess::lineBufferReadWriteControl(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& buffer) {
    buffer = input; // 简单复制，可以根据实际需求进行更复杂的操作
}

// 划窗数据读取控制
void Preprocess::slidingWindowReadControl(const std::vector<std::vector<float>>& input, std::vector<std::vector<std::vector<float>>>& windowed_output) {
    int height = input.size();
    int width = input[0].size();
    int window_size = 3; // 例如3x3窗口
    for (int i = 0; i <= height - window_size; i += window_size) {
        for (int j = 0; j <= width - window_size; j += window_size) {
            std::vector<std::vector<float>> window(window_size, std::vector<float>(window_size, 0));
            for (int wi = 0; wi < window_size; ++wi) {
                for (int wj = 0; wj < window_size; ++wj) {
                    window[wi][wj] = input[i + wi][j + wj];
                }
            }
            windowed_output.push_back(window);
        }
    }
}

// 数据格式变换
void Preprocess::dataFormatTransform(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& transformed_output) {
    int height = input.size();
    int width = input[0].size();
    transformed_output.resize(width, std::vector<float>(height, 0));
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            transformed_output[j][i] = input[i][j];
        }
    }
}
