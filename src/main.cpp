#include <iostream>

#include <opencv2/core/cuda.hpp>

int main() {
  const int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
  std::cout << "CUDA-enabled OpenCV devices: " << deviceCount << std::endl;
  return 0;
}
