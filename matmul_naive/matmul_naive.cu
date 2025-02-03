
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

class Matrix {
public:
    Matrix() = default;
    Matrix(size_t h, size_t w) : height(h), width(w), elements(nullptr) {}
    Matrix(size_t h, size_t w, float* ele) : height(h), width(w), elements(ele) {}

    size_t height{0};
    size_t width{0};
    float* elements{nullptr};
};

// matmul kernel
__global__ void MatMulKernel(const Matrix A, const Matrix B, const Matrix C) {
    // Each thread computes one element of C
    // by accumulating results into cvalue
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < C.height && col < C.width) {
        // Kahan summation formula
        float cvalue = 0;
        float loss = 0;
        for (size_t k = 0; k < A.width; ++k) {
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
    size_t M = 1000;
    size_t N = 2000;
    size_t K = 1000;

    double gflops = 2.0 * M * N * K * 1.0e-9;

    Matrix hostA(M, K);
    Matrix hostB(K, N);
    Matrix hostC(M, N);

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
    Matrix devA(M, K);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&devA.elements), sizeA));
    checkCudaErrors(cudaMemcpy(devA.elements, hostA.elements, sizeA, cudaMemcpyHostToDevice));

    Matrix devB(K, N);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&devB.elements), sizeB));
    checkCudaErrors(cudaMemcpy(devB.elements, hostB.elements, sizeB, cudaMemcpyHostToDevice));

    Matrix devC(M, N);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&devC.elements), sizeC));

    // launch the matmul cuda kernel
    int nIter = 300;
    std::cout << "Execute the kernel for " << nIter << " iters.\n";

    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((devC.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (devC.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
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
                correct = false;
            }
        }
    }

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
