cmake_minimum_required(VERSION 3.23)
project(vectorAddDrv CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} vectorAdd_kernel.cu vectorAddDrv.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME}
        cuda
)