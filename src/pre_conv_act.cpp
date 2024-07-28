#include "pre_conv_act.h"
#include "preprocess.h"
#include "convolution.h"
#include "activation.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

struct ConvParams {
    std::vector<int> input_size;
    std::vector<int> kernel_size;
    int padding;
    int stride;
    ActivationType activation;
    std::vector<std::vector<std::vector<float>>> input_data;
    std::vector<std::vector<float>> filters;
    std::vector<float> biases;
};

ConvParams readJson(const std::string& filename) {
    std::ifstream file(filename);
    nlohmann::json jsonData;
    file >> jsonData;

    ConvParams params;
    params.input_size = jsonData["input_size"].get<std::vector<int>>();
    params.kernel_size = jsonData["kernel_size"].get<std::vector<int>>();
    params.padding = jsonData["padding"];
    params.stride = jsonData["stride"];
    std::string act_type = jsonData["activation"];
    if (act_type == "sigmoid") {
        params.activation = ActivationType::SIGMOID;
    } else {
        params.activation = ActivationType::NONE;
    }
    params.input_data = jsonData["input_data"].get<std::vector<std::vector<std::vector<float>>>>();
    params.filters = jsonData["filters"].get<std::vector<std::vector<float>>>();
    params.biases = jsonData["biases"].get<std::vector<float>>();

    return params;
}

void pre_conv_act(const std::string& config_file) {
    ConvParams params = readJson(config_file);

    Preprocess preprocess(params.padding, params.kernel_size[2], params.stride);
    std::vector<std::vector<float>> img2col_output;
    preprocess.img2col(params.input_data, img2col_output);

    Convolution convolution(params.input_size[1], params.kernel_size[0], params.kernel_size[2], params.stride, params.padding);
    std::vector<std::vector<float>> output;
    convolution.execute(img2col_output, params.filters, output, params.biases);

    int output_size = (params.input_size[2] - params.kernel_size[2] + 2 * params.padding) / params.stride + 1;
    std::vector<std::vector<std::vector<float>>> feature_maps(params.kernel_size[0], 
        std::vector<std::vector<float>>(output_size, std::vector<float>(output_size, 0)));
    for (int f = 0; f < params.kernel_size[0]; ++f) {
        for (int i = 0; i < output_size * output_size; ++i) {
            feature_maps[f][i / output_size][i % output_size] = output[f][i];
        }
    }

    Activation activation;
    activation.setType(params.activation);
    activation.apply(feature_maps);

    std::cout << "Activated Feature Maps:" << std::endl;
    for (int f = 0; f < feature_maps.size(); ++f) {
        std::cout << "Feature Map " << f << ":\n";
        for (const auto& row : feature_maps[f]) {
            for (float val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}
