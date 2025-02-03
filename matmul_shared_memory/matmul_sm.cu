/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 */

#include "cmdline.h"
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
struct Matrix {
    Matrix() = default;
    Matrix(size_t h, size_t w, size_t s) : height(h), width(w), stride(s) {}
    Matrix(size_t h, size_t w, size_t s, float* ele) : height(h), width(w), stride(s), elements(ele) {}

    size_t height{0};
    size_t width{0};
    size_t stride{0};
    float* elements{nullptr};
};

// Get a matrix element
__device__ float GetElement(const Matrix& A, size_t row, size_t col) {
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(const Matrix& A, size_t row, size_t col, float val) {
    A.elements[row * A.stride + col] = val;
}

// Get a sub-matrix of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(const Matrix& A, size_t row, size_t col, size_t height, size_t width) {
    Matrix sub;
    sub.height = height;
    sub.width = width;
    sub.stride = A.stride;
    sub.elements = &A.elements[row * blockDim.y * A.stride + col * blockDim.x];
    return sub;
}

// Matrix multiplication kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // block row and col
    size_t blockRow = blockIdx.y;
    size_t blockCol = blockIdx.x;

    size_t heightTail = C.height % blockDim.y;
    size_t widthTail = C.width % blockDim.x;

    size_t tileM = blockIdx.y == gridDim.y - 1 && heightTail != 0 ? heightTail : blockDim.y;
    size_t tileN = blockIdx.x == gridDim.x - 1 && widthTail != 0 ? widthTail : blockDim.x;

    // thread row and col within csub
    size_t row = threadIdx.y;
    size_t col = threadIdx.x;

    // each thread block computes one sub-matrix of C
    auto csub = GetSubMatrix(C, blockRow, blockCol, tileM, tileN);

    // each thread computes one element of csub
    // by accumulating results into cvalue.
    // kahan summation formula
    float cvalue = 0;
    float loss = 0;

    // loop over all the sub-matrices of A and B
    // are required to compute csub.
    // Multipy each pair of sub-matrices together
    // and accumulate the results.
    size_t blockNum = (A.width + blockSize - 1) / blockSize;
    size_t tail = A.width % blockSize;
    for (int m = 0; m < blockNum; m++) {
        size_t tileK = m == blockNum - 1 && tail != 0 ? tail : blockSize;

        // Get subA and subB
        auto subA = GetSubMatrix(A, blockRow, m, tileM, tileK);
        auto subB = GetSubMatrix(B, m, blockCol, tileK, tileN);

        // Allocate shared memory used to store subA and subB
        __shared__ float smA[blockSize][blockSize];
        __shared__ float smB[blockSize][blockSize];

        // Load subA and subB from global memory to
        // shared memory. Each thread loads one element
        // of each sub-matrix.
        if (row < tileM && col < tileK) {
            smA[row][col] = GetElement(subA, row, col);
        } else {
            smA[row][col] = 0;
        }

        if (row < tileK && col < tileN) {
            smB[row][col] = GetElement(subB, row, col);
        } else {
            smB[row][col] = 0;
        }

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation.
        __syncthreads();

        // multiply subA and subB
        float sum = 0;
        float loss1 = 0;

#pragma unroll
        for (int k = 0; k < tileK; k++) {
            float cur = smA[row][k] * smB[k][col] - loss1;
            float tmp = sum + cur;
            loss1 = tmp - sum - cur;
            sum = tmp;
        }

        float cur = sum - loss;
        float tmp = cvalue + cur;
        loss = tmp - cvalue - cur;
        cvalue = tmp;

        __syncthreads();
    }

    // write csub to global memory
    // each thread writes one element
    if (blockIdx.y * blockDim.y + threadIdx.y < C.height &&
        blockIdx.x * blockDim.x + threadIdx.x < C.width) {
        SetElement(csub, row, col, cvalue);
    }
}

// Host code
int main(int argc, char** argv) {
    // load A and B to device memory
    std::cout << "[Matrix Multiply Using CUDA] - Starting...\n";
    size_t M = 1000;
    size_t N = 2000;
    size_t K = 1000;

    double gflops = 2.0 * M * N * K * 1.0e-9;

    Matrix hostA(M, K, K);
    Matrix hostB(K, N, N);
    Matrix hostC(M, N, N);

    // Allocate host memory for matrix A
    size_t sizeA = hostA.height * hostA.width * sizeof(float);
    checkCudaErrors(cudaMallocHost(&hostA.elements, sizeA));

    // Allocate host memory for matrix B
    size_t sizeB = hostB.height * hostB.width * sizeof(float);
    checkCudaErrors(cudaMallocHost(&hostB.elements, sizeB));

    // Allocate host memory for matrix C
    size_t sizeC = hostC.height * hostC.width * sizeof(float);
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
    Matrix devA(M, K, K);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&devA.elements), sizeA));
    checkCudaErrors(cudaMemcpy(devA.elements, hostA.elements, sizeA, cudaMemcpyHostToDevice));

    Matrix devB(K, N, N);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&devB.elements), sizeB));
    checkCudaErrors(cudaMemcpy(devB.elements, hostB.elements, sizeB, cudaMemcpyHostToDevice));

    Matrix devC(M, N, N);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&devC.elements), sizeC));

    // launch the matmul cuda kernel
    int nIter = 300;
    std::cout << "Execute the kernel for " << nIter << " iters.\n";

    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((devC.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (devC.height + threadsPerBlock.x - 1) / threadsPerBlock.y);
    double run_time = 0;
    for (int i = 0; i < nIter; ++i) {
        Timer t;
        MatMulKernel<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC);
        double tmp = t.GetElapsedTime();
        run_time = i == 0 ? tmp : std::min(run_time, tmp);
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
                std::cout << "host: " << c << ", "
                          << "device: " << hostC.elements[i * hostC.width + j] << std::endl;
                std::cout << std::endl;
                correct = false;
            }
        }
    }

    std::cout << "runtime = " << run_time << std::endl;
    std::cout << "gflops = " << gflops / run_time << std::endl;

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