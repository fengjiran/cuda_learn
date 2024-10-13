cmake_minimum_required(VERSION 3.23)
project(matmul_naive CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} matmul_naive.cu)
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
