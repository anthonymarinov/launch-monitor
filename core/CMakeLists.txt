add_library(lm_core
  src/example.cpp
)

add_library(launch_monitor::core ALIAS lm_core)

target_include_directories(lm_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(lm_core PUBLIC cxx_std_23)