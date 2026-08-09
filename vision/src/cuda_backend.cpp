#include <launch_monitor/vision/main.hpp>
#include <opencv2/core/cuda.hpp>

int launch_monitor::vision::cuda_device_count() {
  return cv::cuda::getCudaEnabledDeviceCount();
}