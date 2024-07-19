// preprocess.h
#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>

// 定义预处理类
class Preprocess {
public:
    // 构造函数，初始化归一化因子和目标图像尺寸
    Preprocess(float scaleFactor, const cv::Size& newSize);
    
    // 归一化函数，将图像像素值转换到[0, 1]范围
    void normalize(cv::Mat& image) const;
    
    // 图像缩放函数，将图像调整到指定尺寸
    void resize(cv::Mat& image) const;
    
    // 预处理函数，先缩放图像，再进行归一化
    void preprocess(const cv::Mat& input, cv::Mat& output) const;

private:
    float scaleFactor_;  // 归一化因子
    cv::Size newSize_;   // 目标图像尺寸
};

#endif // PREPROCESS_H
