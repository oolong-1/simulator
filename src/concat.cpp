// src/concat.cpp
#include "../include/concat.h"
#include <stdexcept>

Concat::Concat() {}

// 获取连接轴的索引
int Concat::getConcatAxisIndex(int axis, const std::vector<std::vector<std::vector<float>>>& tensor) const {
    int num_axes = tensor.size();
    if (axis < 0 || axis >= num_axes) {
        throw std::invalid_argument("Axis out of bounds.");
    }
    return axis;
}

// 执行连接操作
void Concat::concatenate(const std::vector<std::vector<std::vector<float>>>& input1,
                         const std::vector<std::vector<std::vector<float>>>& input2,
                         std::vector<std::vector<std::vector<float>>>& output,
                         int axis) const {
    if (input1.size() != input2.size() || input1[0].size() != input2[0].size()) {
        throw std::invalid_argument("Input dimensions must match, except for the concatenation axis.");
    }

    int concat_axis = getConcatAxisIndex(axis, input1);

    output = input1;

    for (size_t i = 0; i < input2.size(); ++i) {
        for (size_t j = 0; j < input2[i].size(); ++j) {
            output[i][j].insert(output[i][j].end(), input2[i][j].begin(), input2[i][j].end());
        }
    }
}
