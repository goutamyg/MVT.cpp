cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-fPIC -std=c++17")
set(CMAKE_C_FLAGS "-fPIC -std=c11  -O3 -mavx512f")

project(mvt-onnxruntime-demo)

set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_BUILD_TYPE Debug) # Set the build type to "Debug"
set(CMAKE_BUILD_TYPE "Release")

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(mvt_demo main.cpp mvt.cpp)
# add_executable(test test.cpp)

target_include_directories(mvt_demo PRIVATE ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.12.1/include)
target_compile_features(mvt_demo PRIVATE cxx_std_17)
target_link_libraries(mvt_demo ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.12.1/lib/libonnxruntime.so)
