add_executable(launch_monitor src/main.cpp)

target_link_libraries(launch_monitor PRIVATE
  launch_monitor::vision
  launch_monitor::physics
)