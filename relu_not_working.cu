// relu_forward.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void relu_forward(const float* in, float* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] > 0.0f ? in[idx] : 0.0f;
    }
}

int main() {
    const int N = 10;
    float h_in[N]  = {-3, -1, 0, 1, 2, -5, 6, -2, 4, -7};
    float h_out[N];

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    relu_forward<<<gridSize, blockSize>>>(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < N; i++) {
        printf("%f -> %f\n", h_in[i], h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
