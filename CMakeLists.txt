cmake_minimum_required(VERSION 3.20)
project(LaunchMonitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LAUNCH_MONITOR_ENABLE_CUDA "Enable the CUDA vision backend" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(core)
add_subdirectory(vision)
add_subdirectory(physics)
add_subdirectory(apps/launch_monitor)
