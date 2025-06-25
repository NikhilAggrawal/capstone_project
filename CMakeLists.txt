cmake_minimum_required(VERSION 3.18)
project(cuda_nn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Specify CUDA architectures (optional)
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
add_executable(cuda_nn
    src/main.cu
    src/cuda_kernels.cu
    src/neural_net.cu
    src/utils.cpp
)

set_target_properties(cuda_nn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
