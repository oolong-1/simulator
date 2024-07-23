// include/adder_tree.h
#ifndef ADDER_TREE_H
#define ADDER_TREE_H

#include <vector>

class AdderTree {
public:
    float sum(const std::vector<float>& input) const;

private:
    float adderTree(const std::vector<float>& input) const;
};

#endif // ADDER_TREE_H
