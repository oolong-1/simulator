#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include "../npu/pre.h"
#include "../npu/cim.h"
#include "../npu/vpu.h"
#include "../run/read.h"

using json = nlohmann::json;

int main() {
    std::vector<std::vector<std::vector<float>>> img = {
        {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
        {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}},
        {{19, 20, 21}, {22, 23, 24}, {25, 26, 27}}
    };
    std::vector<std::vector<float>> col_output;
    std::vector<std::vector<std::vector<float>>> pad_output;
    std::vector<std::vector<float>> flattened_filters;
    std::vector<std::vector<std::vector<float>>> conv_output;
    std::vector<float> biases;
    std::vector<std::vector<std::vector<float>>> pool_output;

    // 打开JSON文件
    std::ifstream file("input/op.json");
    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }

    std::cout << "文件成功打开" << std::endl;

    // 解析JSON文件
    json data;
    file >> data;
    std::cout << "文件成功解析" << std::endl;

	// 遍历op

    for (const auto& config : data) {
        int op = config.at("op");
        int pre = config.at("pre");
        int Img2col = config.at("Img2col");
        int pad = config.at("pad");
        int cim = config.at("cim");
        int conv = config.at("conv");
        int fc = config.at("fc");
        int vpu = config.at("vpu");
        int act = config.at("act");
        int pool = config.at("pool");

        if (pre == 1) {
            if (pad == 1) {
                int pad_value = config.at("pad_value");
                padControl(img, pad_output, pad_value);
                img = pad_output;
            }
            if (Img2col == 1) {
                int kernel_size = config.at("kernel_size");
                int kernel_stride = config.at("kernel_stride");
                img2col(img, col_output, kernel_size, kernel_stride);
            }
        }
        if (cim == 1) {
            if (conv == 1) {
                std::vector<std::vector<std::vector<std::vector<float>>>> filters;
                filters = config.at("filters").get<std::vector<std::vector<std::vector<std::vector<float>>>>>();
                biases = config.at("biases").get<std::vector<float>>();
                flattenFilters(filters, flattened_filters);
                executeConvolution(col_output, flattened_filters, conv_output, biases);
                img = conv_output;    
            }
            if (fc == 1) {
                // 添加全连接层处理逻辑（如果有）
            }
        }
        if (vpu == 1) {
            if (act == 1) {
                std::string act_type = config.at("act_type");
                applyActivation(img, act_type);
            }
            if (pool == 1) {
                // 设置池化参数
                std::string pool_type = config.at("pool_type");
                int pool_size = config.at("pool_size");
                int pool_stride = config.at("pool_stride");
                executePooling(img, pool_output, pool_type, pool_size, pool_stride);
                img = pool_output;
            }
        }
    }

    file.close();

    return 0;
}
