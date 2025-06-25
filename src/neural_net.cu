#include <cstdlib>
#include <cuda_runtime.h>
#include "neural_net.h"

void initialize_weights(float* d_W1, float* d_W2, int in_size, int hidden_size, int out_size) {
    float* h_W1 = new float[in_size * hidden_size];
    float* h_W2 = new float[hidden_size * out_size];

    for (int i = 0; i < in_size * hidden_size; i++)
        h_W1[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < hidden_size * out_size; i++)
        h_W2[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;

    cudaMemcpy(d_W1, h_W1, in_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, hidden_size * out_size * sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_W1;
    delete[] h_W2;
}
