
/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of global memory.
 */

#include "helper_cuda.h"
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

// matmul kernel
__global__ void MatMulKernel(const Matrix A, const Matrix B, const Matrix C) {
    // Each thread computes one element of C
    // by accumulating results into cvalue
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < C.height && col < C.width) {
        // Kahan summation formula
        float cvalue = 0;
        float loss = 0;
        for (int k = 0; k < A.width; ++k) {
            float cur = A.elements[row * A.width + k] * B.elements[k * B.width + col] - loss;
            float tmp = cvalue + cur;
            loss = tmp - cvalue - cur;
            cvalue = tmp;
        }
        C.elements[row * C.width + col] = cvalue;
    }
}

int main(int argc, char** argv) {
    constexpr int blockSize = 16;// Thread block size

    // Error code to check return values for CUDA calls
    // cudaError_t err = cudaSuccess;

    std::cout << "[Matrix Multiply Using CUDA] - Starting...\n";
    Matrix hostA;
    hostA.width = 100 * blockSize;
    hostA.height = 100 * blockSize;

    Matrix hostB;
    hostB.width = 200 * blockSize;
    hostB.height = 100 * blockSize;

    Matrix hostC;
    hostC.width = hostB.width;
    hostC.height = hostA.height;

    // Allocate host memory for matrix A
    unsigned int sizeA = hostA.height * hostA.width * sizeof(float);
    checkCudaErrors(cudaMallocHost(&hostA.elements, sizeA));

    // Allocate host memory for matrix B
    unsigned int sizeB = hostB.height * hostB.width * sizeof(float);
    checkCudaErrors(cudaMallocHost(&hostB.elements, sizeB));

    // Allocate host memory for matrix C
    unsigned int sizeC = hostC.height * hostC.width * sizeof(float);
    checkCudaErrors(cudaMallocHost(&hostC.elements, sizeC));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    // initialize host memory A
    for (int i = 0; i < hostA.height * hostA.width; ++i) {
        hostA.elements[i] = dist(gen);
    }

    // initialize host memory B
    for (int i = 0; i < hostB.height * hostB.width; ++i) {
        hostB.elements[i] = dist(gen);
    }

    // Allocate device memory for matrix A
    Matrix devA(hostA);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&devA.elements), sizeA));
    checkCudaErrors(cudaMemcpy(devA.elements, hostA.elements, sizeA, cudaMemcpyHostToDevice));

    Matrix devB(hostB);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&devB.elements), sizeB));
    checkCudaErrors(cudaMemcpy(devB.elements, hostB.elements, sizeB, cudaMemcpyHostToDevice));

    Matrix devC(hostC);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&devC.elements), sizeC));

    // launch the matmul cuda kernel
    int nIter = 300;
    std::cout << "Execute the kernel for " << nIter << "\n";

    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid(devC.width / threadsPerBlock.x, devC.height / threadsPerBlock.y);
    for (int i = 0; i < nIter; ++i) {
        MatMulKernel<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC);
    }

    checkCudaErrors(cudaGetLastError());

    // Copy the device result matrix in device memory to the host result vector
    // in host memory.
    std::cout << "Copy output data from the CUDA device to the host memory\n";
    checkCudaErrors(cudaMemcpy(hostC.elements, devC.elements, sizeC, cudaMemcpyDeviceToHost));

    // verify that the result matrix is correct
    std::cout << "Checking computed result for correctness...\n";
    bool correct = true;

    for (int i = 0; i < hostC.height; ++i) {
        for (int j = 0; j < hostC.width; ++j) {
            double c = 0;
            for (int k = 0; k < hostA.width; ++k) {
                float a = hostA.elements[i * hostA.width + k];
                float b = hostB.elements[k * hostB.width + j];
                c += a * b;
            }
            if (std::fabs((c - hostC.elements[i * hostC.width + j]) / c) > std::numeric_limits<float>::epsilon()) {
                std::cerr << "Result verification failed at element ("
                          << i << ", " << j << ")" << std::endl;
                correct = false;
            }
        }
    }

    // free host memory
    checkCudaErrors(cudaFreeHost(hostA.elements));
    hostA.elements = nullptr;

    checkCudaErrors(cudaFreeHost(hostB.elements));
    hostB.elements = nullptr;

    checkCudaErrors(cudaFreeHost(hostC.elements));
    hostC.elements = nullptr;

    // free device memory
    checkCudaErrors(cudaFree(devA.elements));
    checkCudaErrors(cudaFree(devB.elements));
    checkCudaErrors(cudaFree(devC.elements));

    if (!correct) {
        exit(EXIT_FAILURE);
    }

    std::cout << "Done\n";

    return 0;
}
