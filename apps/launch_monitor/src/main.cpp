#include <iostream>
#include <launch_monitor/vision/main.hpp>

int main() {
  std::cout << "CUDA devices: "
            << launch_monitor::vision::cuda_device_count()
            << '\n';
}
