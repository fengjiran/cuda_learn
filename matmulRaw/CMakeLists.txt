cmake_minimum_required(VERSION 3.23)
project(matmulRaw CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} matmulRaw.cu)
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
