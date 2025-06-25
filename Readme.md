# CUDA MNIST Neural Network

## Description
This project implements a simple Feedforward Neural Network running entirely on the GPU using CUDA. The network is designed to perform inference (forward pass) on the MNIST dataset of handwritten digits (0–9).

The purpose of this capstone project is to demonstrate:
- GPU memory management
- CUDA kernel programming
- Parallel matrix-vector operations

> Note: This project only performs forward inference with **randomly initialized weights**, meaning no learning or training is involved. As expected, the accuracy will be around random chance (~10%).

Requirements:
- CUDA Toolkit (e.g., 12.x)
- Visual Studio with C++ build tools (on Windows) or GCC/Clang on Linux
- CMake (version ≥ 3.22)

## Build Instructions
```bash
mkdir build
cd build
cmake ..
cmake --build .
