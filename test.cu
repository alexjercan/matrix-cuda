#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    printf("Device %d: %s\n", device, props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);

    return 0;
}
