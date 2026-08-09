cmake_minimum_required(VERSION 3.18)
project(LaunchMonitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CUDAToolkit_ROOT "/opt/cuda" CACHE PATH "CUDA Toolkit root")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(my_program src/main.cpp)

target_link_libraries(my_program PRIVATE
  CUDA::cudart
  opencv_core
  opencv_cudaarithm
)
