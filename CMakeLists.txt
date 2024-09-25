cmake_minimum_required(VERSION 3.23)
project(cuda_learn)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit)

include_directories(${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/common
        ${CUDAToolkit_INCLUDE_DIRS}
        "/home/richard/project/cutlass-3.5.1/include")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(vectorAdd)
add_subdirectory(vectorAddDrv)
