// src/convolution.cpp
#include "../include/convolution.h"
#include <iostream>

Convolution::Convolution() : cim_array_(3) {} // 假设CIM阵列大小为3

void Convolution::setParams(const Conv2DParams& params) {
    params_ = params;
}

void Convolution::execute(const std::vector<std::vector<std::vector<float>>>& input,
                          const std::vector<std::vector<std::vector<std::vector<float>>>>& weights,
                          const std::vector<float>& biases,
                          std::vector<std::vector<std::vector<float>>>& output) const {
    int output_height = (input[0].size() - params_.kernel_size + 2 * params_.padding) / params_.stride + 1;
    int output_width = (input[0][0].size() - params_.kernel_size + 2 * params_.padding) / params_.stride + 1;

    output.resize(params_.output_channels, std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0)));

    for (int oc = 0; oc < params_.output_channels; ++oc) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                std::vector<float> sum(params_.input_channels * params_.kernel_size * params_.kernel_size, 0);

                for (int ic = 0; ic < params_.input_channels; ++ic) {
                    for (int kh = 0; kh < params_.kernel_size; ++kh) {
                        for (int kw = 0; kw < params_.kernel_size; ++kw) {
                            int ih = oh * params_.stride + kh - params_.padding;
                            int iw = ow * params_.stride + kw - params_.padding;

                            if (ih >= 0 && ih < input[ic].size() && iw >= 0 && iw < input[ic][0].size()) {
                                cim_array_.loadVector(input[ic][ih]);
                                std::vector<float> weight_vec = weights[oc][ic][kh];
                                cim_array_.loadMatrix({weight_vec});
                                sum.push_back(adder_tree_.sum(cim_array_.multiply()));
                            }
                        }
                    }
                }

                output[oc][oh][ow] = adder_tree_.sum(sum) + biases[oc];
            }
        }
    }
}
