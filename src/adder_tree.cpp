// src/adder_tree.cpp
#include "../include/adder_tree.h"
#include <algorithm>
#include <cmath>

float AdderTree::sum(const std::vector<float>& input) const {
    return adderTree(input);
}

float AdderTree::adderTree(const std::vector<float>& input) const {
    if (input.size() == 1) {
        return input[0];
    }

    std::vector<float> next_level;
    for (size_t i = 0; i < input.size(); i += 2) {
        if (i + 1 < input.size()) {
            next_level.push_back(input[i] + input[i + 1]);
        } else {
            next_level.push_back(input[i]);
        }
    }
    return adderTree(next_level);
}
