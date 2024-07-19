// preprocess.cpp
#include "../include/preprocess.h"

// 构造函数，实现初始化
Preprocess::Preprocess(float scaleFactor, const cv::Size& newSize)
    : scaleFactor_(scaleFactor), newSize_(newSize) {}

// 归一化函数，将图像像素值转换到[0, 1]范围
void Preprocess::normalize(cv::Mat& image) const {
    // convertTo函数将图像转换为浮点型，并使用scaleFactor进行缩放
    image.convertTo(image, CV_32F, scaleFactor_);
}

// 图像缩放函数，将图像调整到指定尺寸
void Preprocess::resize(cv::Mat& image) const {
    // resize函数将图像调整到newSize_大小
    cv::resize(image, image, newSize_);
}

// 预处理函数，先缩放图像，再进行归一化
void Preprocess::preprocess(const cv::Mat& input, cv::Mat& output) const {
    // 复制输入图像到输出图像
    output = input.clone();
    
    // 调整图像尺寸
    resize(output);
    
    // 进行归一化
    normalize(output);
}
