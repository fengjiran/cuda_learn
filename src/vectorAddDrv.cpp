//
// Created by richard on 9/24/24.
//

// Includes
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <iostream>

// includes, project
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>

// includes, CUDA
#include <builtin_types.h>

// Variables
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction vecAdd_kernel;

float* h_A;
float* h_B;
float* h_C;

CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;

// Functions
int CleanupNoFailure();
void RandomInit(float*, int);
bool findModulePath(const char*, std::string&, char**, std::string&);

// define input fatbin file
#ifndef FATBIN_FILE
#define FATBIN_FILE "vectorAdd_kernel64.fatbin"
#endif

// Host code
int test_vectorAddDrv(int argc, char** argv) {
    std::cout << "Vector Addition (Driver API)\n";
    int N = 50000;
    int devID = 0;
    size_t size = N * sizeof(float);

    // Initialize
    checkCudaErrors(cuInit(0));
    cuDevice = findCudaDeviceDRV(argc, const_cast<const char**>(argv));

    // Create context
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

    // First search for the module path before we load the results
    std::string module_path;
    std::ostringstream fatbin;



    return 0;
}