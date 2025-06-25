#include <cmath>
#include "utils.h"
#include <iostream>
#include <fstream>
#include <vector>

void softmax(float* vec, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        vec[i] = expf(vec[i]);
        sum += vec[i];
    }
    for (int i = 0; i < size; i++) {
        vec[i] /= sum;
    }
}

bool load_mnist_images(const std::string& filename, std::vector<float>& images, int& num_images, int image_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open " << filename << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t filesize = file.tellg();
    file.seekg(0, std::ios::beg);

    num_images = filesize / image_size;
    images.resize(num_images * image_size);

    std::vector<unsigned char> buffer(num_images * image_size);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    for (size_t i = 0; i < buffer.size(); i++) {
        images[i] = buffer[i] / 255.0f;  // Normalize to 0-1
    }

    return true;
}

bool load_mnist_labels(const std::string& filename, std::vector<int>& labels, int& num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open " << filename << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t filesize = file.tellg();
    file.seekg(0, std::ios::beg);

    num_labels = filesize;
    labels.resize(num_labels);

    std::vector<unsigned char> buffer(num_labels);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    for (size_t i = 0; i < buffer.size(); i++) {
        labels[i] = static_cast<int>(buffer[i]);
    }

    return true;
}