#pragma once
#include <filesystem>

namespace launch_monitor::vision {
  int cuda_device_count();
  bool frame_difference(const std::filesystem::path& input_path,
                        const std::filesystem::path& output_path);
}
