#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void sigmoidActivation(float *z, float *act){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    act[idx] = 1.0f / (1.0f + expf(-z[idx]));
}

int main() {
    int arraySize = 10;
    float h_z[10] = {1,2,3,4,5,6,7,8,9,10};
    float h_act[10];

    float *device_z_values, *device_activations;
    cudaMalloc(&device_z_values, arraySize * sizeof(float));
    cudaMalloc(&device_activations, arraySize * sizeof(float));

    cudaMemcpy(device_z_values, h_z, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch (must be inside main)
    sigmoidActivation<<<1, arraySize>>>(device_z_values, device_activations);

    cudaDeviceSynchronize(); // ensure kernel is done

    cudaMemcpy(h_act, device_activations, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; i++) {
        printf("%f -> %f\n", h_z[i], h_act[i]);
    }

    cudaFree(device_z_values);
    cudaFree(device_activations);

    return 0;
}
