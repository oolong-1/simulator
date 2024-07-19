// include/batch_normalization.h
#ifndef BATCH_NORMALIZATION_H
#define BATCH_NORMALIZATION_H

#include <vector>

class BatchNormalization {
public:
    BatchNormalization();
    void setParams(float epsilon, const std::vector<float>& gamma, const std::vector<float>& beta);
    void execute(const std::vector<std::vector<std::vector<float>>>& input,
                 std::vector<std::vector<std::vector<float>>>& output) const;

private:
    float epsilon_; // 防止除以零的小常数
    std::vector<float> gamma_; // 缩放参数
    std::vector<float> beta_; // 平移参数
};

#endif // BATCH_NORMALIZATION_H
