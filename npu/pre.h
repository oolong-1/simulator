#ifndef PRE_H
#define PRE_H

#include <vector>

// img2col 函数声明：将输入的三维向量转换为列向量
void img2col(const std::vector<std::vector<std::vector<float>>>& img_input,
             std::vector<std::vector<float>>& col_output,
             int kernel_size, int kernel_stride);

// flattenFilters 函数声明：将四维过滤器向量展平成二维向量
void flattenFilters(const std::vector<std::vector<std::vector<std::vector<float>>>>& filters,
                    std::vector<std::vector<float>>& flattened_filters);

// padControl 函数声明：对输入的三维向量进行填充
void padControl(const std::vector<std::vector<std::vector<float>>>& pad_input,
                std::vector<std::vector<std::vector<float>>>& pad_output,
                int pad_value);

#endif // PRE_H
