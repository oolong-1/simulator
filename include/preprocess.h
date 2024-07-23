// include/preprocess.h
#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <vector>

class Preprocess {
public:
    Preprocess(int tile_size, int pad_value);
    void tileReadControl(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& tiled_output);
    void padControl(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& padded_output);
    void lineBufferReadWriteControl(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& buffer);
    void slidingWindowReadControl(const std::vector<std::vector<float>>& input, std::vector<std::vector<std::vector<float>>>& windowed_output);
    void dataFormatTransform(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& transformed_output);

private:
    int tile_size_;
    int pad_value_;
};

#endif // PREPROCESS_H
