// include/convolution.h
#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>
#include "cim_array.h"
#include "adder_tree.h"

struct Conv2DParams {
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
};

class Convolution {
public:
    Convolution();
    void setParams(const Conv2DParams& params);
    void execute(const std::vector<std::vector<std::vector<float>>>& input,
                 const std::vector<std::vector<std::vector<std::vector<float>>>>& weights,
                 const std::vector<float>& biases,
                 std::vector<std::vector<std::vector<float>>>& output) const;

private:
    Conv2DParams params_;
    mutable CIMArray cim_array_;
    mutable AdderTree adder_tree_;
};

#endif // CONVOLUTION_H
