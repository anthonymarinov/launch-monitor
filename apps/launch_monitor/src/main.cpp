#include <iostream>
#include <launch_monitor/vision/main.hpp>

int main() {
  return launch_monitor::vision::frame_difference(
      "data/swing.mov", "output/tracking.mp4")
      ? 0
      : 1;
}
