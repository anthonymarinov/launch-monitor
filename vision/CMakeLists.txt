add_library(lm_vision
  src/ball_detector.cpp
  src/pipeline.cpp
)

add_library(launch_monitor::vision ALIAS lm_vision)

target_include_directories(lm_vision
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(lm_vision PUBLIC launch_monitor::core)

find_package(OpenCV REQUIRED COMPONENTS
  core
  geometry
  imgproc
  videoio
)
target_link_libraries(lm_vision PRIVATE
  opencv_core
  opencv_geometry
  opencv_imgproc
  opencv_videoio
)

if(LAUNCH_MONITOR_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(OpenCV REQUIRED COMPONENTS cudaarithm cudaimgproc)

  target_sources(lm_vision PRIVATE src/cuda_backend.cpp)
  target_link_libraries(lm_vision PRIVATE
    CUDA::cudart
    opencv_cudaarithm
    opencv_cudaimgproc
  )
  target_compile_definitions(lm_vision PRIVATE LAUNCH_MONITOR_HAS_CUDA=1)
else()
  target_sources(lm_vision PRIVATE src/cpu_backend.cpp)
endif()
