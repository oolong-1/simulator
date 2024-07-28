#include "../include/preprocess.h"
#include "../include/convolution.h"
#include "../include/activation.h"
#include <iostream>

int main() {
    // 输入数据
    std::vector<std::vector<std::vector<float>>> input_data = {
        {
            {1, 2, 1},
            {0, 1, 0},
            {1, 2, 1}
        },
        {
            {1, 0, 1},
            {2, 1, 2},
            {1, 0, 1}
        },
        {
            {2, 2, 2},
            {1, 1, 1},
            {0, 0, 0}
        }
    };

    // 卷积核权重
    std::vector<std::vector<float>> filters = {
        {1, 0, 0, -1,  1, 0, 0, -1,  1, 0, 0, -1},  // 卷积核1
        {-1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1}    // 卷积核2
    };

    // 偏置
    std::vector<float> biases = {1, -1};  // 对于两个卷积核

    // 前处理：没有填充，步长为1
    Preprocess preprocess(0, 2, 1);  // 填充值0, 卷积核大小2, 步长1
    std::vector<std::vector<float>> img2col_output;
    preprocess.img2col(input_data, img2col_output);

    // 打印img2col的输出以检查其正确性
    std::cout << "img2col output size: " << img2col_output.size() << " x " << img2col_output[0].size() << std::endl;

    // 卷积
    Convolution convolution(3, 2, 2, 1, 0);
    std::vector<std::vector<float>> output;
    convolution.execute(img2col_output, filters, output, biases);

    // 将卷积输出转换为特征图形状
    std::vector<std::vector<std::vector<float>>> feature_maps(2, std::vector<std::vector<float>>(2, std::vector<float>(2, 0)));
    for (int f = 0; f < 2; ++f) {
        for (int i = 0; i < 4; ++i) {
            feature_maps[f][i / 2][i % 2] = output[f][i];
        }
    }

    // 应用sigmoid激活函数
    Activation activation;
    activation.setType(ActivationType::SIGMOID);
    activation.apply(feature_maps);

    // 打印激活后的输出特征图
    std::cout << "Activated Feature Maps:" << std::endl;
    for (int f = 0; f < feature_maps.size(); ++f) {
        std::cout << "Feature Map " << f << ":\n";
        for (const auto& row : feature_maps[f]) {
            for (float val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}
