#pragma once

#include <cstdint>
#include <optional>

#include <opencv2/core.hpp>

namespace launch_monitor::vision {

struct BallObservation {
  std::int64_t frame_index{};
  double timestamp_ms{};
  cv::Point2f center{};
  float radius_px{};
  float confidence{};
};

struct BallDetectorConfig {
  // Normalized x, y, width, and height values. This is a default tee area for
  // the current camera setup, not an absolute pixel region. It should be
  // adjusted for the device's physical capture volume during setup.
  cv::Rect2f pre_shot_search_region{0.25F, 0.55F, 0.25F, 0.20F};
  float min_radius_px{35.0F};
  float max_radius_px{80.0F};
  float min_brightness{120.0F};
  // Use zero for a neutral-white ball. The light-blue test ball benefits from
  // a positive minimum, which rejects desaturated concrete false positives.
  float min_saturation{45.0F};
  float max_saturation{200.0F};
};

class BallDetector {
 public:
  explicit BallDetector(BallDetectorConfig config = {});

  [[nodiscard]] std::optional<BallObservation> locate_stationary_ball(
      const cv::Mat& frame,
      std::int64_t frame_index = 0,
      double timestamp_ms = 0.0) const;

 private:
  BallDetectorConfig config_;
};

}  // namespace launch_monitor::vision
