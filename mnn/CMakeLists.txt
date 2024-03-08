cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-fPIC -std=c++17")
set(CMAKE_C_FLAGS "-fPIC -std=c11  -O3 -mavx512f")

project(mvt-mnn-demo)

set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_BUILD_TYPE Debug) # Set the build type to "Debug"
set(CMAKE_BUILD_TYPE "Release")

# Find OpenCV package
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/third_party/MNN/include)

link_directories(/usr/lib/x86_64-linux-gnu ${PROJECT_SOURCE_DIR}/third_party/MNN/build)

# add_executable(mvt_demo main.cpp mvt_mnn.cpp)
add_executable(mvt_demo main.cpp mvt.cpp)

# target_link_libraries(mvt_demo libMNN.so ${OpenCV_LIBS})
target_link_libraries(mvt_demo libMNN.so ${OpenCV_LIBS})
