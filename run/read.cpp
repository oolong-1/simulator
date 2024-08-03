#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include "read.h"

// 使用 nlohmann::json
using json = nlohmann::json;

void readimg() {
    const std::string input_filename = "input/xiaoxin.jpg";
    const std::string output_filename = "output/input.json";

    // 读取图片
    cv::Mat image = cv::imread(input_filename);
    if (image.empty()) {
        std::cerr << "无法打开图片" << std::endl;
        return;
    }

    // 显示读取的图像
    cv::imshow("Read Image", image);
    cv::waitKey(0); // 等待用户按键关闭窗口

    // 将图像缩小到3x3
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(3, 3), 0, 0, cv::INTER_AREA);

    // 初始化输入向量
    std::vector<std::vector<std::vector<float>>> input(3, std::vector<std::vector<float>>(3, std::vector<float>(3, 0.0f)));

    // 遍历压缩后的图像并将像素值存储到向量中
    for (int row = 0; row < resized_image.rows; ++row) {
        for (int col = 0; col < resized_image.cols; ++col) {
            cv::Vec3b pixel = resized_image.at<cv::Vec3b>(row, col);
            input[0][row][col] = static_cast<float>(pixel[0]); // 蓝色通道
            input[1][row][col] = static_cast<float>(pixel[1]); // 绿色通道
            input[2][row][col] = static_cast<float>(pixel[2]); // 红色通道
        }
    }

    // 将输入向量转换为json对象
    json j;
    for (size_t channel = 0; channel < input.size(); ++channel) {
        for (size_t row = 0; row < input[channel].size(); ++row) {
            for (size_t col = 0; col < input[channel][row].size(); ++col) {
                j["channel_" + std::to_string(channel)][row][col] = input[channel][row][col];
            }
        }
    }

    // 将json对象保存到文件
    std::ofstream file(output_filename);
    if (file.is_open()) {
        file << j.dump(4); // 格式化输出，缩进4个空格
        file.close();
    } else {
        std::cerr << "无法打开输出文件" << std::endl;
    }
}
