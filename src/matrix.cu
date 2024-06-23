#define NUM_THREADS_PER_BLOCK 32

__global__ void matrixMulKernel(float *d_A, float *d_B, float *d_C, int N, int M, int P) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= M) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < P; i++) {
        sum += d_A[row * P + i] * d_B[i * M + col];
    }

    d_C[row * M + col] = sum;
}

extern "C" {
#include <assert.h>
#include "matrix.h"
#include "ds.h"

void matrix_matmul(const Matrix *m1, const Matrix *m2, Matrix *matrix) {
    assert(m1->cols == m2->rows);
    assert(matrix->rows == m1->rows);
    assert(matrix->cols == m2->cols);
    assert(matrix->data != NULL);

    cudaError_t error;
    float *d_A, *d_B, *d_C;

    error = cudaMalloc((void **)&d_A, m1->rows * m1->cols * sizeof(float));
    if (error != cudaSuccess) {
        DS_PANIC("cudaMalloc failed: %s\n", cudaGetErrorString(error));
    }

    error = cudaMalloc((void **)&d_B, m2->rows * m2->cols * sizeof(float));
    if (error != cudaSuccess) {
        DS_PANIC("cudaMalloc failed: %s\n", cudaGetErrorString(error));
    }

    error = cudaMalloc((void **)&d_C, matrix->rows * matrix->cols * sizeof(float));
    if (error != cudaSuccess) {
        DS_PANIC("cudaMalloc failed: %s\n", cudaGetErrorString(error));
    }

    error = cudaMemcpy(d_A, m1->data, m1->rows * m1->cols * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        DS_PANIC("cudaMemcpy failed: %s\n", cudaGetErrorString(error));
    }

    error = cudaMemcpy(d_B, m2->data, m2->rows * m2->cols * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        DS_PANIC("cudaMemcpy failed: %s\n", cudaGetErrorString(error));
    }

    dim3 dimGrid((matrix->rows + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK,
                 (matrix->cols + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK);
    dim3 dimBlock(NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK);

    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, matrix->rows, matrix->cols, m1->cols);
    cudaDeviceSynchronize();

    error = cudaMemcpy(matrix->data, d_C, matrix->rows * matrix->cols * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        DS_PANIC("cudaMemcpy failed: %s\n", cudaGetErrorString(error));
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
}
