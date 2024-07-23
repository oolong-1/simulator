cmake_minimum_required(VERSION 3.10)
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

# 链接OpenCV库
target_link_libraries(main ${OpenCV_LIBS})
