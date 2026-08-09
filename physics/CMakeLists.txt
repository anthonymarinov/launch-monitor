add_library(lm_physics
  src/example.cpp
)

add_library(launch_monitor::physics ALIAS lm_physics)

target_include_directories(lm_physics
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(lm_physics PUBLIC cxx_std_23)

target_link_libraries(lm_physics PRIVATE launch_monitor::core)