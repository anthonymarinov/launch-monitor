#pragma once

#include <launch_monitor/vision/ball_tracker.hpp>

namespace launch_monitor::vision {

struct BallKalmanTrackerConfig {
  float direct_measurement_variance{0.01F};
  float recovered_measurement_variance{1600.0F};
  float position_process_variance{4.0F};
  float velocity_process_variance{10000.0F};
  float initial_state_variance{100.0F};
};

class BallKalmanTracker {
 public:
  explicit BallKalmanTracker(BallKalmanTrackerConfig config = {});

  [[nodiscard]] BallTrack smooth(const BallTrack& raw_track) const;

 private:
  BallKalmanTrackerConfig config_;
};

}  // namespace launch_monitor::vision
