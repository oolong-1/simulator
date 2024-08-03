#ifndef VPU_H
#define VPU_H

#include <vector>
#include <string>

// 激活函数声明
void applyActivation(std::vector<std::vector<std::vector<float>>>& input, const std::string& type);

// 最大池化函数声明
void maxPooling(const std::vector<std::vector<std::vector<float>>>& input,
                std::vector<std::vector<std::vector<float>>>& output,
                int pool_size, int pool_stride);

// 平均池化函数声明
void averagePooling(const std::vector<std::vector<std::vector<float>>>& input,
                    std::vector<std::vector<std::vector<float>>>& output,
                    int pool_size, int pool_stride);

// 池化函数声明
void executePooling(const std::vector<std::vector<std::vector<float>>>& input,
                    std::vector<std::vector<std::vector<float>>>& output,
                    const std::string& pool_type, int pool_size, int pool_stride);

#endif // VPU_H
