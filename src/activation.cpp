// src/activation.cpp
#include "../include/activation.h"
#include <algorithm>
#include <cmath>

// 默认构造函数
Activation::Activation() : type_(ActivationType::NONE) {}

// 设置激活函数类型
void Activation::setType(ActivationType type) {
    type_ = type;
}

// 应用激活函数
void Activation::apply(std::vector<std::vector<std::vector<float>>>& input) const {
    for (auto& channel : input) {
        for (auto& row : channel) {
            for (auto& val : row) {
                switch (type_) {
                    case ActivationType::RELU:
                        val = std::max(0.0f, val);
                        break;
                    case ActivationType::LEAKY_RELU:
                        val = std::max(0.01f * val, val);
                        break;
                    case ActivationType::SIGMOID:
                        val = 1.0f / (1.0f + std::exp(-val));
                        break;
                    case ActivationType::TANH:
                        val = std::tanh(val);
                        break;
                    case ActivationType::NONE:
                    default:
                        break;
                }
            }
        }
    }
}
