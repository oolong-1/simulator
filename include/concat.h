// include/concat.h
#ifndef CONCAT_H
#define CONCAT_H

#include <vector>

class Concat {
public:
    Concat();
    void concatenate(const std::vector<std::vector<std::vector<float>>>& input1,
                     const std::vector<std::vector<std::vector<float>>>& input2,
                     std::vector<std::vector<std::vector<float>>>& output,
                     int axis) const;

private:
    int getConcatAxisIndex(int axis, const std::vector<std::vector<std::vector<float>>>& tensor) const;
};

#endif // CONCAT_H
