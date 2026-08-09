#include <launch_monitor/vision/ball_kalman_tracker.hpp>

#include <cmath>
#include <vector>

#include <opencv2/video/tracking.hpp>

namespace {

void set_transition(cv::KalmanFilter& filter, float elapsed_seconds) {
  filter.transitionMatrix = cv::Mat::eye(4, 4, CV_32F);
  filter.transitionMatrix.at<float>(0, 2) = elapsed_seconds;
  filter.transitionMatrix.at<float>(1, 3) = elapsed_seconds;
}

cv::KalmanFilter make_filter(const launch_monitor::vision::BallKalmanTrackerConfig& config,
                             const cv::Point2f& center,
                             const cv::Point2f& velocity) {
  cv::KalmanFilter filter{4, 2, 0, CV_32F};
  set_transition(filter, 0.0F);
  filter.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
  filter.measurementMatrix.at<float>(0, 0) = 1.0F;
  filter.measurementMatrix.at<float>(1, 1) = 1.0F;
  filter.processNoiseCov = cv::Mat::zeros(4, 4, CV_32F);
  filter.processNoiseCov.at<float>(0, 0) = config.position_process_variance;
  filter.processNoiseCov.at<float>(1, 1) = config.position_process_variance;
  filter.processNoiseCov.at<float>(2, 2) = config.velocity_process_variance;
  filter.processNoiseCov.at<float>(3, 3) = config.velocity_process_variance;
  cv::setIdentity(filter.errorCovPost, cv::Scalar{config.initial_state_variance});
  filter.statePost = cv::Mat::zeros(4, 1, CV_32F);
  filter.statePost.at<float>(0) = center.x;
  filter.statePost.at<float>(1) = center.y;
  filter.statePost.at<float>(2) = velocity.x;
  filter.statePost.at<float>(3) = velocity.y;
  return filter;
}

void set_measurement_variance(cv::KalmanFilter& filter, float variance) {
  cv::setIdentity(filter.measurementNoiseCov, cv::Scalar{variance});
}

cv::Mat point_measurement(const cv::Point2f& center) {
  cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F);
  measurement.at<float>(0) = center.x;
  measurement.at<float>(1) = center.y;
  return measurement;
}

cv::Point2f point_from_state(const cv::Mat& state) {
  return {state.at<float>(0), state.at<float>(1)};
}

cv::Point2f velocity_between(const launch_monitor::vision::TrackedBallObservation& first,
                             const launch_monitor::vision::TrackedBallObservation& second) {
  const double elapsed_seconds =
      (second.observation.timestamp_ms - first.observation.timestamp_ms) / 1000.0;
  if (elapsed_seconds <= 0.0) {
    return {};
  }
  return (second.observation.center - first.observation.center) *
         static_cast<float>(1.0 / elapsed_seconds);
}

}  // namespace

namespace launch_monitor::vision {

BallKalmanTracker::BallKalmanTracker(BallKalmanTrackerConfig config) : config_(config) {}

BallTrack BallKalmanTracker::smooth(const BallTrack& raw_track) const {
  BallTrack result = raw_track;
  if (result.points.size() < 2) {
    return result;
  }

  for (TrackedBallObservation& point : result.points) {
    point.measured_center_px = point.observation.center;
  }

  std::vector<size_t> direct_indices;
  for (size_t index = 0; index < result.points.size(); ++index) {
    if (!result.points[index].observation.recovered) {
      direct_indices.push_back(index);
    }
  }
  if (direct_indices.size() < 2) {
    return result;
  }

  const size_t first_direct_index = direct_indices.front();
  const size_t second_direct_index = direct_indices[1];
  const cv::Point2f initial_velocity = velocity_between(
      result.points[first_direct_index], result.points[second_direct_index]);

  cv::KalmanFilter forward = make_filter(
      config_, result.points[first_direct_index].observation.center, initial_velocity);
  double last_timestamp_ms = result.points[first_direct_index].observation.timestamp_ms;

  for (size_t direct_offset = 1; direct_offset < direct_indices.size(); ++direct_offset) {
    TrackedBallObservation& point = result.points[direct_indices[direct_offset]];
    const float elapsed_seconds = static_cast<float>(
        (point.observation.timestamp_ms - last_timestamp_ms) / 1000.0);
    if (elapsed_seconds <= 0.0F) {
      continue;
    }

    set_transition(forward, elapsed_seconds);
    forward.predict();
    set_measurement_variance(forward, config_.direct_measurement_variance);
    const cv::Mat corrected = forward.correct(point_measurement(point.measured_center_px));
    point.observation.center = point_from_state(corrected);
    last_timestamp_ms = point.observation.timestamp_ms;
  }

  // A standard forward filter cannot revise a point before initialization.
  // Run the same constant-velocity model backward from the first direct point
  // and give recovered measurements a much larger uncertainty.
  cv::KalmanFilter backward = make_filter(
      config_, result.points[first_direct_index].observation.center, initial_velocity);
  double next_timestamp_ms = result.points[first_direct_index].observation.timestamp_ms;
  for (size_t index = first_direct_index; index-- > 0;) {
    TrackedBallObservation& point = result.points[index];
    const float elapsed_seconds = static_cast<float>(
        (point.observation.timestamp_ms - next_timestamp_ms) / 1000.0);
    set_transition(backward, elapsed_seconds);
    backward.predict();
    set_measurement_variance(backward,
                             point.observation.recovered
                                 ? config_.recovered_measurement_variance
                                 : config_.direct_measurement_variance);
    const cv::Mat corrected = backward.correct(point_measurement(point.measured_center_px));
    point.observation.center = point_from_state(corrected);
    next_timestamp_ms = point.observation.timestamp_ms;
  }

  for (size_t index = 0; index < result.points.size(); ++index) {
    cv::Point2f velocity;
    if (index == 0) {
      velocity = velocity_between(result.points[index], result.points[index + 1]);
    } else if (index + 1 == result.points.size()) {
      velocity = velocity_between(result.points[index - 1], result.points[index]);
    } else {
      velocity = velocity_between(result.points[index - 1], result.points[index + 1]);
    }
    result.points[index].velocity_px_per_s = velocity;
    result.points[index].speed_px_per_s = static_cast<float>(cv::norm(velocity));
  }

  result.initial_velocity_px_per_s = result.points.front().velocity_px_per_s;
  const float initial_speed = static_cast<float>(cv::norm(result.initial_velocity_px_per_s));
  if (initial_speed > 0.0F) {
    result.launch_direction = result.initial_velocity_px_per_s * (1.0F / initial_speed);
  }
  return result;
}

}  // namespace launch_monitor::vision
