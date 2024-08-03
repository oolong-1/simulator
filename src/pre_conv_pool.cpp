#include "../include/pre_conv_pool.h"
#include "../include/pre.h"
#include "../include/convolution.h"
#include "../include/pooling.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

struct ConvPoolParams {
    std::vector<int> input_size;
    std::vector<int> kernel_size;
    int padding;
    int stride;
    PoolingParams pooling_params;
    std::vector<std::vector<std::vector<std::vector<float>>>> filters;
    std::vector<std::vector<std::vector<float>>> input_data;
    std::vector<float> biases;
};

ConvPoolParams readJson(const std::string& filename) {
    std::ifstream file(filename);
    nlohmann::json jsonData;
    file >> jsonData;

    ConvPoolParams params;
    params.input_size = jsonData["input_size"].get<std::vector<int>>();
    params.kernel_size = jsonData["kernel_size"].get<std::vector<int>>();
    params.padding = jsonData["padding"];
    params.stride = jsonData["stride"];
    std::string pooling_type = jsonData["pooling_type"];
    if (pooling_type == "max") {
        params.pooling_params.type = PoolingType::MAX;
    } else if (pooling_type == "average") {
        params.pooling_params.type = PoolingType::AVERAGE;
    } else {
        throw std::invalid_argument("Invalid pooling type");
    }
    params.pooling_params.pool_size = jsonData["pool_size"];
    params.pooling_params.pool_stride = jsonData["pool_stride"];
    params.input_data = jsonData["input_data"].get<std::vector<std::vector<std::vector<float>>>>();
    params.filters = jsonData["filters"].get<std::vector<std::vector<std::vector<std::vector<float>>>>>();
    params.biases = jsonData["biases"].get<std::vector<float>>();

    return params;
}

void pre_conv_pool(const std::string& config_file) {
    ConvPoolParams params = readJson(config_file);

    Preprocess preprocess(params.padding, params.kernel_size[2], params.stride);
    std::vector<std::vector<float>> img2col_output;
    preprocess.img2col(params.input_data, img2col_output);

    // Flatten the filters
    std::vector<std::vector<float>> flattened_filters;
    preprocess.flattenFilters(params.filters, flattened_filters);

    Convolution convolution(params.input_size[1], params.kernel_size[0], params.kernel_size[2], params.stride, params.padding);
    std::vector<std::vector<float>> conv_output;
    convolution.execute(img2col_output, flattened_filters, conv_output, params.biases);

    int output_size = (params.input_size[2] - params.kernel_size[2] + 2 * params.padding) / params.stride + 1;
    std::vector<std::vector<std::vector<float>>> feature_maps(params.kernel_size[0], 
        std::vector<std::vector<float>>(output_size, std::vector<float>(output_size, 0)));
    for (int f = 0; f < params.kernel_size[0]; ++f) {
        for (int i = 0; i < output_size * output_size; ++i) {
            feature_maps[f][i / output_size][i % output_size] = conv_output[f][i];
        }
    }

    Pooling pooling;
    pooling.setParams(params.pooling_params);
    std::vector<std::vector<std::vector<float>>> pooled_output;
    pooling.execute(feature_maps, pooled_output);

    std::cout << "Pooled Feature Maps:" << std::endl;
    for (int f = 0; f < pooled_output.size(); ++f) {
        std::cout << "Feature Map " << f << ":\n";
        for (const auto& row : pooled_output[f]) {
            for (float val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}
