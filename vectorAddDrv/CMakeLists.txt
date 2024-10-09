cmake_minimum_required(VERSION 3.23)
project(vectorAddDrv CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

#include_directories(${PROJECT_SOURCE_DIR})

add_executable(${PROJECT_NAME} vectorAdd_kernel.cu vectorAddDrv.cpp)

#target_compile_options(${PROJECT_NAME}
#        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fatbin>
#)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME}
        cuda
)