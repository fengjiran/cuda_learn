//
// Created by richard on 10/11/24.
//

#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#include <cstdio>

// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char* _cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}
#endif

template<typename T>
void check(T result, char const* const func, const char* const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


#endif//HELPER_CUDA_H
