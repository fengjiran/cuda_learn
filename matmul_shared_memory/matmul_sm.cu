/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 */

#include "helper_cuda.h"
#include <cstdio>
#include <cuda_runtime.h>// For the CUDA runtime routines (prefixed with "cuda_")
#include <iostream>
#include <random>
#include <vector>

// Thread block size
constexpr int blockSize = 16;

// Matrices are stored in row-major order
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float val) {
    A.elements[row * A.stride + col] = val;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix sub;
    sub.height = blockSize;
    sub.width = blockSize;
    sub.stride = A.stride;
    sub.elements = &A.elements[row * blockSize * A.stride + col * blockSize];
    return sub;
}

// Matrix multiplication kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // block row and col
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // each thread block computes one sub-matrix of C
    auto csub = GetSubMatrix(C, blockRow, blockCol);

    // each thread computes one element of csub
    // by accumulating results into cvalue
    float cvalue = 0;

    // thread row and col within csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // loop over all the sub-matrices of A and B
    // are required to compute csub.
    // Multipy each pair of sub-matrices together
    // and accumulate the results.
    for (int m = 0; m < A.width / blockSize; m++) {
        // Get subA and subB
        auto subA = GetSubMatrix(A, blockRow, m);
        auto subB = GetSubMatrix(B, m, blockCol);

        // Allocate shared memory used to store subA and subB
        __shared__ float smA[blockSize][blockSize];
        __shared__ float smB[blockSize][blockSize];

        // Load subA and subB from global memory to
        // shared memory. Each thread loads one element
        // of each sub-matrix.
        smA[row][col] = GetElement(subA, row, col);
        smB[row][col] = GetElement(subB, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation.
        __syncthreads();

        // multiply subA and subB
        // float tmp = 0;
        // float y = 0;
        for (int k = 0; k < blockSize; k++) {
            // float r;
            // y -= smA[row][k] * smB[k][col];
            // r = tmp - y;
            // y = (r - tmp) + y;
            // tmp = r;
            cvalue += smA[row][k] * smB[k][col];
        }
        // cvalue += tmp;

        __syncthreads();
    }

    // write csub to global memory
    // each thread writes one element
    SetElement(csub, row, col, cvalue);
}

// Host code
int main(int argc, char** argv) {
    // load A and B to device memory
    std::cout << "[Matrix Multiply Using CUDA] - Starting...\n";
    Matrix hostA;
    hostA.width = 100 * blockSize;
    hostA.height = 100 * blockSize;
    hostA.stride = hostA.width;

    Matrix hostB;
    hostB.width = 200 * blockSize;
    hostB.height = 100 * blockSize;
    hostB.stride = hostB.width;

    Matrix hostC;
    hostC.width = hostB.width;
    hostC.height = hostA.height;
    hostC.stride = hostC.width;

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
                // std::cerr << "Result verification failed at element ("
                //           << i << ", " << j << ")" << std::endl;
                // std::cout << "host: " << c << ", "
                //           << "device: " << hostC.elements[i * hostC.width + j] << std::endl;
                correct = false;
                // break;
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