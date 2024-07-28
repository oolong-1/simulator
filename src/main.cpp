#include <iostream>
#include "../include/pre_conv_act.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>" << std::endl;
        return -1;
    }

    std::string config_file = argv[1];
    pre_conv_act(config_file);

    return 0;
}