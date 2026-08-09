#pragma once

#include <optional>
#include <vector>

#include <launch_monitor/vision/ball_detector.hpp>

namespace launch_monitor::vision {

struct TrackedBallObservation {
  BallObservation observation;
  cv::Point2f velocity_px_per_s{};
  float speed_px_per_s{};
};

struct BallTrack {
  std::vector<TrackedBallObservation> points;
  cv::Point2f launch_direction{};
  cv::Point2f initial_velocity_px_per_s{};
};

struct BallTrackerConfig {
  int minimum_track_points{3};
  int maximum_frame_gap{2};
  float minimum_speed_px_per_s{500.0F};
  float minimum_direction_cosine{0.80F};
};

class BallTracker {
 public:
  explicit BallTracker(BallTrackerConfig config = {});

  [[nodiscard]] BallTrack build_track(
      const std::vector<BallObservation>& candidates) const;

  [[nodiscard]] std::optional<BallObservation> predict_previous_observation(
      const BallTrack& track) const;

 private:
  BallTrackerConfig config_;
};

}  // namespace launch_monitor::vision
