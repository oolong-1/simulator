// include/fully_connected.h
#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <vector>

class FullyConnected {
public:
    FullyConnected();
    void setParams(const std::vector<std::vector<float>>& weights, const std::vector<float>& biases);
    void execute(const std::vector<float>& input, std::vector<float>& output) const;

private:
    std::vector<std::vector<float>> weights_;
    std::vector<float> biases_;
};

#endif // FULLY_CONNECTED_H
