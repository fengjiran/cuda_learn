cmake_minimum_required(VERSION 3.23)

project(cuda_learn CUDA)

set(CMAKE_CUDA_STANDARD 17)

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cu)

include_directories(${PROJECT_SOURCE_DIR}/include
        "/home/richard/project/cutlass-3.5.1/include")

add_executable(${PROJECT_NAME} ${SRC_FILES} main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
