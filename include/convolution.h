// include/convolution.h
#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>

class Convolution {
public:
    Convolution(int input_channels, int output_channels, int kernel_size, int stride, int padding);

    void execute(const std::vector<std::vector<float>>& input_col,
                 const std::vector<std::vector<float>>& filters,
                 std::vector<std::vector<float>>& output,
                 const std::vector<float>& biases);

private:
    int input_channels_;
    int output_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
};

#endif // CONVOLUTION_H
