//
// Created by richard on 9/22/24.
//

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <cstdio>
#include <cuda_runtime.h>// For the CUDA runtime routines (prefixed with "cuda_")
#include <iostream>
#include <random>
#include <vector>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float* a, const float* b, float* c, int numElems) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElems) {
        c[i] = a[i] + b[i] + 0.0f;
    }
}

/**
 * Host test routine
 */
int main() {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    std::cout << "[Vector addition of " << numElements << " elements]\n";

    std::vector<float> hostVecA(numElements);// Allocate the host input vector A
    std::vector<float> hostVecB(numElements);// Allocate the host input vector B
    std::vector<float> hostVecC(numElements);// Allocate the host output vector C

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        hostVecA[i] = dist(gen);
        hostVecB[i] = dist(gen);
    }

    // Allocate the device input vector A
    float* devPtrA = nullptr;
    err = cudaMalloc(reinterpret_cast<void**>(&devPtrA), size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device vector A ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector A
    float* devPtrB = nullptr;
    err = cudaMalloc(reinterpret_cast<void**>(&devPtrB), size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device vector B ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float* devPtrC = nullptr;
    err = cudaMalloc(reinterpret_cast<void**>(&devPtrC), size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device vector C ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input
    // vectors in device memory
    std::cout << "Copy input data from the host memory to the CUDA device\n";
    err = cudaMemcpy(devPtrA, hostVecA.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy vector A from host to device ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(devPtrB, hostVecB.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy vector B from host to device ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    // launch the vector add cuda kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads\n";
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(devPtrA, devPtrB, devPtrC, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cerr << "Failed to launch vectorAdd kernel ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    std::cout << "Copy output data from the CUDA device to the host memory\n";
    err = cudaMemcpy(hostVecC.data(), devPtrC, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy vector C from device to host ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i) {
        if (std::fabs(hostVecA[i] + hostVecB[i] - hostVecC[i]) > std::numeric_limits<float>::epsilon()) {
            std::cerr << "Result verification failed at element " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test PASSED\n";

    // Free device global memory
    err = cudaFree(devPtrA);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free device vector A ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    err = cudaFree(devPtrB);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free device vector B ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }

    err = cudaFree(devPtrC);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free device vector C ("
                  << "error code " << cudaGetErrorString(err) << ")!\n";
        exit(EXIT_FAILURE);
    }


    std::cout << "Done\n";

    return 0;
}