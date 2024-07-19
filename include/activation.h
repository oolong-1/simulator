// include/activation.h
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>

enum class ActivationType {
    NONE,
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH
};

class Activation {
public:
    Activation();
    void setType(ActivationType type);
    void apply(std::vector<std::vector<std::vector<float>>>& input) const;

private:
    ActivationType type_;
};

#endif // ACTIVATION_H
