
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

// matmul kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Each thread computes one element of C
    // by accumulating results into cvalue
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float cvalue = 0;
    float y = 0;
    for (int k = 0; k < A.width; ++k) {
        float r;
        y -= A.elements[row * A.width + k] * B.elements[k * B.width + col];
        r = cvalue - y;
        y = (r - cvalue) + y;
        cvalue = r;
        // cvalue += A.elements[row * A.width + k] * B.elements[k * B.width + col];
    }
    C.elements[row * C.width + col] = cvalue;
}

int main(int argc, char** argv) {
    constexpr int blockSize = 16;// Thread block size

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    std::cout << "[Matrix Multiply Using CUDA] - Starting...\n";
    Matrix hostA;
    hostA.width = 10 * blockSize;
    hostA.height = 10 * blockSize;

    Matrix hostB;
    hostB.width = 20 * blockSize;
    hostB.height = 10 * blockSize;

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

    // Allocate host memory for matrix C
    unsigned int sizeC = hostC.height * hostC.width * sizeof(float);
    err = cudaMallocHost(&hostC.elements, sizeC);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate host memory for matrix C "
                  << "(error code " << cudaGetErrorString(err) << ")\n";
        exit(EXIT_FAILURE);
    }

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
    err = cudaMalloc(reinterpret_cast<void**>(&devA.elements), sizeA);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for matrix A "
                  << "(error code " << cudaGetErrorString(err) << ")\n";
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(devA.elements, hostA.elements, sizeA, cudaMemcpyHostToDevice);

    Matrix devB(hostB);
    err = cudaMalloc(reinterpret_cast<void**>(&devB.elements), sizeB);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for matrix B "
                  << "(error code " << cudaGetErrorString(err) << ")\n";
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(devB.elements, hostB.elements, sizeB, cudaMemcpyHostToDevice);

    Matrix devC(hostC);
    err = cudaMalloc(reinterpret_cast<void**>(&devC.elements), sizeC);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for matrix C "
                  << "(error code " << cudaGetErrorString(err) << ")\n";
        exit(EXIT_FAILURE);
    }

    // launch the matmul cuda kernel
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid(devC.width / threadsPerBlock.x, devC.height / threadsPerBlock.y);
    MatMulKernel<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch Matmul kernel ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    // Copy the device result matrix in device memory to the host result vector
    // in host memory.
    std::cout << "Copy output data from the CUDA device to the host memory\n";
    err = cudaMemcpy(hostC.elements, devC.elements, sizeC, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy matrix C from device to host ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    // verify that the result matrix is correct
    for (int i = 0; i < hostC.height; ++i) {
        for (int j = 0; j < hostC.width; ++j) {
            double c = 0;
            for (int k = 0; k < hostA.width; ++k) {
                float a = hostA.elements[i * hostA.width + k];
                float b = hostB.elements[k * hostB.width + j];
                c += a * b;
            }
            // std::cout << "device value = " << c << std::endl;
            // std::cout << "host value = " << hostC.elements[i * hostC.width + j] << std::endl;
            // std::cout << "fabs = " << std::fabs((c - hostC.elements[i * hostC.width + j]) / c) << std::endl;
            // std::cout << "epsilon = " << std::numeric_limits<float>::epsilon() << std::endl;
            // std::cout << std::endl;
            if (std::fabs((c - hostC.elements[i * hostC.width + j]) / c) > std::numeric_limits<float>::epsilon()) {
                std::cerr << "Result verification failed at element ("
                          << i << ", " << j << ")" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    // free host memory
    err = cudaFreeHost(hostA.elements);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free host matrix A ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }
    hostA.elements = nullptr;

    err = cudaFreeHost(hostB.elements);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free host matrix B ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }
    hostB.elements = nullptr;

    err = cudaFreeHost(hostC.elements);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free host matrix C ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }
    hostC.elements = nullptr;

    // free device memory
    err = cudaFree(devA.elements);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free device matrix A ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    err = cudaFree(devB.elements);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free device matrix B ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    err = cudaFree(devC.elements);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free host matrix C ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "Done\n";

    return 0;
}
