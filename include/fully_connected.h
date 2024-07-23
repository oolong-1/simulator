// include/fully_connected.h
#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <vector>
#include "cim_array.h"
#include "adder_tree.h"

class FullyConnected {
public:
    FullyConnected(int size);
    void setParams(const std::vector<std::vector<float>>& weights, const std::vector<float>& biases);
    void execute(const std::vector<float>& input, std::vector<float>& output) const;

private:
    int size_;
    mutable CIMArray cim_array_;
    mutable AdderTree adder_tree_;
    std::vector<std::vector<float>> weights_;
    std::vector<float> biases_;
};

#endif // FULLY_CONNECTED_H
