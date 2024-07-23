// main.cpp
#include "include/convolution.h"
#include "include/activation.h"
#include "include/pooling.h"
#include "include/batch_normalization.h"
#include "include/fully_connected.h"
#include "include/preprocess.h"
#include <iostream>
#include <vector>

// 打印矩阵函数，用于显示图像的像素值
void printTensor(const std::vector<std::vector<std::vector<float>>>& tensor) {
    for (const auto& channel : tensor) {
        for (const auto& row : channel) {
            for (float val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// 打印向量函数，用于显示全连接层输出
void printVector(const std::vector<float>& vec) {
    for (float val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    // 定义卷积层参数
    Conv2DParams conv_params = {3, 1, 3, 1, 1}; // 输入通道数、输出通道数、卷积核大小、步长、填充
    Convolution conv;
    conv.setParams(conv_params);

    // 定义卷积层权重和偏置
    std::vector<std::vector<std::vector<std::vector<float>>>> weights = {
        {
            {
                {1, 0, -1},
                {1, 0, -1},
                {1, 0, -1}
            },
            {
                {0, 1, 0},
                {0, 1, 0},
                {0, 1, 0}
            },
            {
                {-1, 0, 1},
                {-1, 0, 1},
                {-1, 0, 1}
            }
        }
    };

    std::vector<float> biases = {0};

    // 定义激活函数
    Activation activation;
    activation.setType(ActivationType::LEAKY_RELU);

    // 定义池化参数
    PoolingParams pooling_params = {PoolingType::MAX, 2, 2};
    Pooling pooling;
    pooling.setParams(pooling_params);

    // 定义批量归一化参数
    std::vector<float> gamma = {1.0f}; // 通道数应为out_channels的大小
    std::vector<float> beta = {0.0f};  // 通道数应为out_channels的大小
    BatchNormalization batch_norm;
    batch_norm.setParams(1e-5, gamma, beta);

    // 定义全连接层参数
    std::vector<std::vector<float>> fc_weights = {
        {0.1f, 0.2f, 0.3f, 0.4f},
        {0.5f, 0.6f, 0.7f, 0.8f},
        {0.9f, 1.0f, 1.1f, 1.2f}
    };
    std::vector<float> fc_biases = {0.1f, 0.2f, 0.3f};
    FullyConnected fully_connected(3);
    fully_connected.setParams(fc_weights, fc_biases);

    // 定义前处理模块
    Preprocess preprocess(2, 1); // 例如tile_size为2，pad_value为1

    // 输入数据
    std::vector<std::vector<std::vector<float>>> input = {
        {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}
        },
        {
            {1, 1, 1, 1},
            {1, 1, 1, 1},
            {1, 1, 1, 1},
            {1, 1, 1, 1}
        },
        {
            {0, 1, 0, 1},
            {1, 0, 1, 0},
            {0, 1, 0, 1},
            {1, 0, 1, 0}
        }
    };

    // 前处理：tile读取控制
    std::vector<std::vector<float>> tiled_output;
    preprocess.tileReadControl(input[0], tiled_output);

    // 前处理：pad控制
    std::vector<std::vector<float>> padded_output;
    preprocess.padControl(tiled_output, padded_output);

    // 前处理：line buffer读写控制
    std::vector<std::vector<float>> buffer_output;
    preprocess.lineBufferReadWriteControl(padded_output, buffer_output);

    // 前处理：划窗数据读取控制
    std::vector<std::vector<std::vector<float>>> windowed_output;
    preprocess.slidingWindowReadControl(buffer_output, windowed_output);

    // 前处理：数据格式变换
    std::vector<std::vector<float>> transformed_output;
    preprocess.dataFormatTransform(buffer_output, transformed_output);

    // 执行卷积操作
    std::vector<std::vector<std::vector<float>>> conv_output;
    conv.execute(input, weights, biases, conv_output);

    // 应用激活函数
    activation.apply(conv_output);

    // 执行池化操作
    std::vector<std::vector<std::vector<float>>> pooled_output;
    pooling.execute(conv_output, pooled_output);

    // 执行批量归一化
    std::vector<std::vector<std::vector<float>>> normalized_output;
    batch_norm.execute(pooled_output, normalized_output);

    // 将三维特征图展平成一维向量
    std::vector<float> flattened_output;
    for (const auto& channel : normalized_output) {
        for (const auto& row : channel) {
            flattened_output.insert(flattened_output.end(), row.begin(), row.end());
        }
    }

    // 执行全连接层
    std::vector<float> fc_output;
    fully_connected.execute(flattened_output, fc_output);

    // 打印全连接层输出
    printVector(fc_output);

    return 0;
}
