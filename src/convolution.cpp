// src/convolution.cpp
#include "../include/convolution.h"
#include <algorithm>

// 默认构造函数
Convolution::Convolution() {}

// 设置卷积参数
void Convolution::setParams(const Conv2DParams& params) {
    params_ = params;
}

// 执行卷积操作
void Convolution::execute(const std::vector<std::vector<std::vector<float>>>& input,
                          const std::vector<std::vector<std::vector<std::vector<float>>>>& weights,
                          const std::vector<float>& biases,
                          std::vector<std::vector<std::vector<float>>>& output) const {
    int in_channels = params_.in_channels;
    int out_channels = params_.out_channels;
    int kernel_size = params_.kernel_size;
    int stride = params_.stride;
    int padding = params_.padding;

    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int output_height = (input_height - kernel_size + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_size + 2 * padding) / stride + 1;

    output.resize(out_channels, std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0)));

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                float value = biases[oc];
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                value += input[ic][ih][iw] * weights[oc][ic][kh][kw];
                            }
                        }
                    }
                }
                output[oc][oh][ow] = value;
            }
        }
    }
}
