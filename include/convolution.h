// include/convolution.h
#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>

// 定义卷积参数结构
struct Conv2DParams {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
};

// 定义卷积类
class Convolution {
public:
    Convolution();
    void setParams(const Conv2DParams& params);
    void execute(const std::vector<std::vector<std::vector<float>>>& input,
                 const std::vector<std::vector<std::vector<std::vector<float>>>>& weights,
                 const std::vector<float>& biases,
                 std::vector<std::vector<std::vector<float>>>& output) const;

private:
    Conv2DParams params_;
};

#endif // CONVOLUTION_H

