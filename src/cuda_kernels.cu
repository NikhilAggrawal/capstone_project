#include <cuda_runtime.h>
#include "cuda_kernels.h"

// Matrix-vector multiplication kernel
__global__ void matvec(const float* mat, const float* vec, float* result, int rows, int cols) {
    int row = threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum += mat[row * cols + i] * vec[i];
        }
        result[row] = sum;
    }
}

// ReLU activation
__global__ void relu(float* vec, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        if (vec[idx] < 0) vec[idx] = 0;
    }
}

void forward_pass(float* input, float* W1, float* W2, float* hidden, float* output,
                  int in_size, int hidden_size, int out_size) {
    // Hidden layer
    matvec<<<1, hidden_size>>>(W1, input, hidden, hidden_size, in_size);
    relu<<<1, hidden_size>>>(hidden, hidden_size);

    // Output layer
    matvec<<<1, out_size>>>(W2, hidden, output, out_size, hidden_size);

    
}
