// include/pooling.h
#ifndef POOLING_H
#define POOLING_H

#include <vector>

enum class PoolingType {
    MAX,
    AVERAGE
};

struct PoolingParams {
    PoolingType type;
    int pool_size;
    int pool_stride;
};

class Pooling {
public:
    Pooling();
    void setParams(const PoolingParams& params);
    void execute(const std::vector<std::vector<std::vector<float>>>& input,
                 std::vector<std::vector<std::vector<float>>>& output) const;

private:
    PoolingParams params_;

    // 最大池化
    void maxPooling(const std::vector<std::vector<std::vector<float>>>& input,
                    std::vector<std::vector<std::vector<float>>>& output) const;

    // 平均池化
    void averagePooling(const std::vector<std::vector<std::vector<float>>>& input,
                        std::vector<std::vector<std::vector<float>>>& output) const;
};

#endif // POOLING_H
