// src/cim_array.cpp
#include "../include/cim_array.h"

CIMArray::CIMArray(int size) : size_(size) {
    matrix_.resize(size_, std::vector<float>(size_, 0));
    vector_.resize(size_, 0);
}

void CIMArray::loadMatrix(const std::vector<std::vector<float>>& matrix) {
    matrix_ = matrix;
}

void CIMArray::loadVector(const std::vector<float>& vector) {
    vector_ = vector;
}

std::vector<float> CIMArray::multiply() const {
    std::vector<float> result(size_, 0);
    for (int i = 0; i < size_; ++i) {
        for (int j = 0; j < size_; ++j) {
            result[i] += matrix_[i][j] * vector_[j];
        }
    }
    return result;
}
