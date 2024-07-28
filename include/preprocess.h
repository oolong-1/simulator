// include/preprocess.h
#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <vector>

class Preprocess {
public:
    // 构造函数：初始化填充值、卷积核大小和步长
    Preprocess(int pad_value, int kernel_size, int stride);

    // 填充控制：为输入数据添加边缘填充
    void padControl(const std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<std::vector<float>>>& padded_output);

    // img2col功能：将输入数据转换为列形式以便进行卷积计算
    void img2col(const std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<float>>& col_output);

private:
    int pad_value_;    // 填充值
    int kernel_size_;  // 卷积核大小
    int stride_;       // 滑动窗口的步长
};

#endif // PREPROCESS_H
