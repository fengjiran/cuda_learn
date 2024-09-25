cmake_minimum_required(VERSION 3.23)

project(cuda_learn CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit)

message(${CMAKE_CURRENT_SOURCE_DIR})
message(${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cu
        ${PROJECT_SOURCE_DIR}/src/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/common
        ${CUDAToolkit_INCLUDE_DIRS}
        "/home/richard/project/cutlass-3.5.1/include")

add_executable(${PROJECT_NAME} ${SRC_FILES} main.cu)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME}
        cuda
)
