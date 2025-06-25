#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#include "neural_net.h"
#include "cuda_kernels.h"
#include "utils.h"

int main() {
    std::cout << "CUDA Neural Network Inference on Test Dataset" << std::endl;

    // Load MNIST test data
    std::vector<float> test_images;
    std::vector<int> test_labels;
    int num_test_images, num_test_labels;

    if (load_mnist_images("../data/mnist_test_images.bin", test_images, num_test_images, 784)) {
        std::cout << "Loaded " << num_test_images << " test images" << std::endl;
    } else {
        std::cerr << "Failed to load test images!" << std::endl;
        return -1;
    }

    if (load_mnist_labels("../data/mnist_test_labels.bin", test_labels, num_test_labels)) {
        std::cout << "Loaded " << num_test_labels << " test labels" << std::endl;
    } else {
        std::cerr << "Failed to load test labels!" << std::endl;
        return -1;
    }

    const int INPUT_SIZE = 784;
    const int HIDDEN_SIZE = 128;
    const int OUTPUT_SIZE = 10;

    // Allocate GPU memory
    float *d_W1, *d_W2, *d_hidden, *d_output, *d_input;
    cudaMalloc(&d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));

    // Initialize weights (random, so accuracy will be low for untrained model)
    initialize_weights(d_W1, d_W2, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    int correct = 0;

    // Loop over all test images
    for (int img_idx = 0; img_idx < num_test_images; img_idx++) {
        // Copy one image to device
        cudaMemcpy(d_input, &test_images[img_idx * INPUT_SIZE], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        // Forward pass
        forward_pass(d_input, d_W1, d_W2, d_hidden, d_output, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

        // Copy output back
        float output[OUTPUT_SIZE];
        cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        // Apply softmax
        float sum = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output[i] = expf(output[i]);
            sum += output[i];
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output[i] /= sum;
        }

        // Find predicted label
        int predicted = 0;
        float max_val = output[0];
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (output[i] > max_val) {
                max_val = output[i];
                predicted = i;
            }
        }

        // Check correctness
        if (predicted == test_labels[img_idx]) {
            correct++;
        }

        // Optional progress output
        if (img_idx % 1000 == 0) {
            std::cout << "Processed " << img_idx << " images..." << std::endl;
        }
    }

    float accuracy = (correct * 100.0f) / num_test_images;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    // Cleanup
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_input);

    return 0;
}
