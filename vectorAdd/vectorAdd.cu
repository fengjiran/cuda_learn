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

#include "helper_cuda.h"
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
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElems) {
        c[i] = a[i] + b[i] + 0.0f;
    }
}

/**
 * Host test routine
 */
int main() {
    // Error code to check return values for CUDA calls
    // cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    std::cout << "[Vector addition of " << numElements << " elements]\n";

    // Allocate the host input vector A
    std::vector<float> hostVecA = GenRandomMatrix(1, numElements);
    // Allocate the host input vector B
    std::vector<float> hostVecB = GenRandomMatrix(1, numElements);
    // Allocate the host output vector C
    std::vector<float> hostVecC(numElements);

    // Allocate the device input vector A
    float* devPtrA = nullptr;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&devPtrA), size));

    // Allocate the device input vector A
    float* devPtrB = nullptr;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&devPtrB), size));

    // Allocate the device output vector C
    float* devPtrC = nullptr;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&devPtrC), size));

    // Copy the host input vectors A and B in host memory to the device input
    // vectors in device memory
    std::cout << "Copy input data from the host memory to the CUDA device\n";
    checkCudaErrors(cudaMemcpy(devPtrA, hostVecA.data(), size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrB, hostVecB.data(), size, cudaMemcpyHostToDevice));

    // launch the vector add cuda kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads\n";
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(devPtrA, devPtrB, devPtrC, numElements);
    checkCudaErrors(cudaGetLastError());

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    std::cout << "Copy output data from the CUDA device to the host memory\n";
    checkCudaErrors(cudaMemcpy(hostVecC.data(), devPtrC, size, cudaMemcpyDeviceToHost));

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i) {
        if (std::fabs(hostVecA[i] + hostVecB[i] - hostVecC[i]) > std::numeric_limits<float>::epsilon()) {
            std::cerr << "Result verification failed at element " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test PASSED\n";

    // Free device global memory
    checkCudaErrors(cudaFree(devPtrA));
    checkCudaErrors(cudaFree(devPtrB));
    checkCudaErrors(cudaFree(devPtrC));

    std::cout << "Done\n";

    return 0;
}