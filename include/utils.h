#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

void softmax(float* vec, int size);
void relu(float* vec, int size);
void relu_derivative(float* vec, int size);
bool load_mnist_images(const std::string& filename, std::vector<float>& images, int& num_images, int image_size);
bool load_mnist_labels(const std::string& filename, std::vector<int>& labels, int& num_labels);

