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
  bool recovered{};
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

  // This direction and cone are relative to the detected tee ball. They define
  // a physical capture volume, not a fixed image rectangle.
  cv::Point2f expected_flight_direction{1.0F, -0.45F};
  float flight_corridor_length_frame_widths{1.20F};
  float flight_corridor_half_angle_degrees{22.0F};
  float min_candidate_area_px{200.0F};
  float max_candidate_area_px{12000.0F};
  float min_candidate_aspect_ratio{0.30F};
  float max_candidate_aspect_ratio{3.20F};
  float min_candidate_radius_tee_ratio{0.65F};
  float max_candidate_radius_tee_ratio{1.60F};
  float min_candidate_confidence{0.65F};
  float min_motion_distance_radii{1.50F};

  float recovery_search_radius_radii{2.50F};
  float recovery_max_center_error_radii{1.15F};
  float recovery_max_radius_ratio{1.35F};
  float recovery_min_brightness{95.0F};
  float recovery_min_saturation{15.0F};
  float recovery_min_confidence{0.35F};
};

class BallDetector {
 public:
  explicit BallDetector(BallDetectorConfig config = {});

  [[nodiscard]] std::optional<BallObservation> locate_stationary_ball(
      const cv::Mat& frame,
      std::int64_t frame_index = 0,
      double timestamp_ms = 0.0) const;

  [[nodiscard]] std::optional<BallObservation> locate_moving_ball(
      const cv::Mat& frame,
      const cv::Mat& motion_mask,
      const BallObservation& tee_ball,
      std::int64_t frame_index,
      double timestamp_ms) const;

  [[nodiscard]] std::optional<BallObservation> locate_near_prediction(
      const cv::Mat& frame,
      const cv::Mat& motion_mask,
      const BallObservation& prediction) const;

 private:
  BallDetectorConfig config_;
};

}  // namespace launch_monitor::vision
