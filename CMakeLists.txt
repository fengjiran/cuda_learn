cmake_minimum_required(VERSION 3.23)

project(cuda_learn CUDA)

set(CMAKE_CUDA_STANDARD 17)
#find_package(GTest REQUIRED)

include_directories("/home/richard/project/cutlass-3.5.1/include")
add_executable(cuda_learn main.cu matmul.cu matmul.cuh common.h utility.h
        vectorAdd.cu)

set_target_properties(cuda_learn PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
