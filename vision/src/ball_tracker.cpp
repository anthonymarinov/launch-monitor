#include <launch_monitor/vision/ball_tracker.hpp>

#include <algorithm>
#include <cmath>
#include <optional>

namespace {

cv::Point2f velocity_between(const launch_monitor::vision::BallObservation& first,
                             const launch_monitor::vision::BallObservation& second) {
  const double elapsed_seconds = (second.timestamp_ms - first.timestamp_ms) / 1000.0;
  if (elapsed_seconds <= 0.0) {
    return {};
  }

  return (second.center - first.center) * static_cast<float>(1.0 / elapsed_seconds);
}

float direction_cosine(const cv::Point2f& first, const cv::Point2f& second) {
  const float first_length = cv::norm(first);
  const float second_length = cv::norm(second);
  if (first_length <= 0.0F || second_length <= 0.0F) {
    return -1.0F;
  }
  return first.dot(second) / (first_length * second_length);
}

}  // namespace

namespace launch_monitor::vision {

BallTracker::BallTracker(BallTrackerConfig config) : config_(config) {}

BallTrack BallTracker::build_track(const std::vector<BallObservation>& candidates) const {
  std::vector<BallObservation> ordered_candidates = candidates;
  std::sort(ordered_candidates.begin(), ordered_candidates.end(),
            [](const BallObservation& left, const BallObservation& right) {
              return left.frame_index < right.frame_index;
            });

  std::vector<BallObservation> current_track;
  std::vector<BallObservation> best_track;
  std::optional<cv::Point2f> prior_velocity;

  const auto retain_if_best = [&best_track](const std::vector<BallObservation>& track) {
    if (track.size() > best_track.size()) {
      best_track = track;
    }
  };

  for (const BallObservation& candidate : ordered_candidates) {
    if (current_track.empty()) {
      current_track.push_back(candidate);
      prior_velocity.reset();
      continue;
    }

    const BallObservation& previous = current_track.back();
    const int frame_gap = static_cast<int>(candidate.frame_index - previous.frame_index);
    const cv::Point2f velocity = velocity_between(previous, candidate);
    const float speed = cv::norm(velocity);
    const bool has_consistent_direction =
        !prior_velocity.has_value() ||
        direction_cosine(*prior_velocity, velocity) >= config_.minimum_direction_cosine;
    const bool belongs_to_track = frame_gap > 0 &&
                                  frame_gap <= config_.maximum_frame_gap &&
                                  speed >= config_.minimum_speed_px_per_s &&
                                  has_consistent_direction;
    if (!belongs_to_track) {
      retain_if_best(current_track);
      current_track = {candidate};
      prior_velocity.reset();
      continue;
    }

    current_track.push_back(candidate);
    prior_velocity = velocity;
  }
  retain_if_best(current_track);

  BallTrack result;
  if (best_track.size() < static_cast<size_t>(config_.minimum_track_points)) {
    return result;
  }

  result.points.reserve(best_track.size());
  for (size_t index = 0; index < best_track.size(); ++index) {
    cv::Point2f velocity;
    if (index == 0) {
      velocity = velocity_between(best_track[index], best_track[index + 1]);
    } else if (index + 1 == best_track.size()) {
      velocity = velocity_between(best_track[index - 1], best_track[index]);
    } else {
      velocity = velocity_between(best_track[index - 1], best_track[index + 1]);
    }

    result.points.push_back(TrackedBallObservation{
        .observation = best_track[index],
        .velocity_px_per_s = velocity,
        .speed_px_per_s = static_cast<float>(cv::norm(velocity)),
    });
  }

  result.initial_velocity_px_per_s = result.points.front().velocity_px_per_s;
  const float direction_length = cv::norm(result.initial_velocity_px_per_s);
  if (direction_length > 0.0F) {
    result.launch_direction = result.initial_velocity_px_per_s *
                              (1.0F / direction_length);
  }
  return result;
}

}  // namespace launch_monitor::vision
