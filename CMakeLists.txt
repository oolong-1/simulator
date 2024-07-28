cmake_minimum_required(VERSION 3.14)  # 修改为 3.14 或更高版本，支持 FetchContent

project(NeuralNetworkModules)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 找到OpenCV库
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

# 添加源文件
file(GLOB SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/main.cpp
)

# 设置可执行文件的输出目录
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

# 添加可执行文件
add_executable(main ${SOURCES})

# 链接OpenCV库和nlohmann/json库
target_link_libraries(main PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2  # 使用的库版本，可以根据需要更改
)

FetchContent_MakeAvailable(json)

target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)
