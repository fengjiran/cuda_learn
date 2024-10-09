
/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of global memory.
 */

#include <cstdio>
#include <cuda_runtime.h>// For the CUDA runtime routines (prefixed with "cuda_")
#include <iostream>
#include <random>
#include <vector>

// Matrices are stored in row-major order
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// matmul kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Each thread computes one element of C
    // by accumulating results into cvalue
    float cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int k = 0; k < A.width; ++k) {
        cvalue += A.elements[row * A.width + k] * B.elements[k * B.width + col];
    }
    C.elements[row * C.width + col] = cvalue;
}

int main(int argc, char** argv) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    std::cout << "[Matrix Multiply Using CUDA] - Starting...\n";
    Matrix hostA;
    hostA.width = 10 * BLOCK_SIZE;
    hostA.height = 10 * BLOCK_SIZE;

    Matrix hostB;
    hostB.width = 20 * BLOCK_SIZE;
    hostB.height = 10 * BLOCK_SIZE;

    Matrix hostC;
    hostC.width = hostB.width;
    hostC.height = hostA.height;

    // Allocate host memory for matrix A
    unsigned int sizeA = hostA.height * hostA.width * sizeof(float);
    err = cudaMallocHost(&hostA.elements, sizeA);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate host memory for matrix A "
                  << "(error code " << cudaGetErrorString(err) << ")\n";
        exit(EXIT_FAILURE);
    }

    // Allocate host memory for matrix B
    unsigned int sizeB = hostB.height * hostB.width * sizeof(float);
    err = cudaMallocHost(&hostB.elements, sizeB);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate host memory for matrix B "
                  << "(error code " << cudaGetErrorString(err) << ")\n";
        exit(EXIT_FAILURE);
    }


    // initialize host memory A
    for (int i = 0; i < hostA.height * hostA.width; ++i) {
        *(hostA.elements + i) = 1.0f;
    }

    // initialize host memory B
    for (int i = 0; i < hostB.height * hostB.width; ++i) {
        *(hostB.elements + i) = 0.01f;
    }


    // free host and device memory
    err = cudaFreeHost(hostA.elements);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free host matrix A ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    err = cudaFreeHost(hostB.elements);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free host matrix B ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    return 0;
}
