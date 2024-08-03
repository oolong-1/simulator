#ifndef CIM_H
#define CIM_H

#include <vector>

// 执行卷积函数声明
void executeConvolution(const std::vector<std::vector<float>>& input,
                        const std::vector<std::vector<float>>& filters,
                        std::vector<std::vector<std::vector<float>>>& output,
                        const std::vector<float>& biases);

#endif // CIM_H
