#include <launch_monitor/vision/ball_detector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/geometry/2d.hpp>
#include <opencv2/imgproc.hpp>

namespace {

cv::Rect normalized_region_to_pixels(const cv::Rect2f& normalized_region,
                                     const cv::Size& frame_size) {
  const float left = std::clamp(normalized_region.x, 0.0F, 1.0F);
  const float top = std::clamp(normalized_region.y, 0.0F, 1.0F);
  const float right = std::clamp(normalized_region.x + normalized_region.width,
                                 0.0F,
                                 1.0F);
  const float bottom = std::clamp(normalized_region.y + normalized_region.height,
                                  0.0F,
                                  1.0F);

  const int x = static_cast<int>(left * frame_size.width);
  const int y = static_cast<int>(top * frame_size.height);
  const int width = static_cast<int>(right * frame_size.width) - x;
  const int height = static_cast<int>(bottom * frame_size.height) - y;
  return {x, y, std::max(width, 0), std::max(height, 0)};
}

float clamp_unit(float value) {
  return std::clamp(value, 0.0F, 1.0F);
}

cv::Mat flight_corridor_mask(const cv::Size& frame_size,
                             const launch_monitor::vision::BallObservation& tee_ball,
                             const launch_monitor::vision::BallDetectorConfig& config) {
  cv::Point2f direction = config.expected_flight_direction;
  const float direction_length = cv::norm(direction);
  if (direction_length <= std::numeric_limits<float>::epsilon()) {
    return {};
  }
  direction *= 1.0F / direction_length;

  const cv::Point2f perpendicular{-direction.y, direction.x};
  const float length = config.flight_corridor_length_frame_widths * frame_size.width;
  const float half_angle_radians =
      config.flight_corridor_half_angle_degrees * static_cast<float>(CV_PI) / 180.0F;
  const float start_half_width = tee_ball.radius_px * 1.2F;
  const float end_half_width =
      start_half_width + length * std::tan(half_angle_radians);
  const cv::Point2f end = tee_ball.center + direction * length;

  const std::array<cv::Point2f, 4> corners{
      tee_ball.center + perpendicular * start_half_width,
      tee_ball.center - perpendicular * start_half_width,
      end - perpendicular * end_half_width,
      end + perpendicular * end_half_width,
  };
  std::array<cv::Point, 4> polygon;
  std::transform(corners.begin(), corners.end(), polygon.begin(), [](const cv::Point2f& point) {
    return cv::Point{static_cast<int>(std::lround(point.x)),
                     static_cast<int>(std::lround(point.y))};
  });

  cv::Mat mask = cv::Mat::zeros(frame_size, CV_8UC1);
  cv::fillConvexPoly(mask, polygon.data(), static_cast<int>(polygon.size()), cv::Scalar{255});
  return mask;
}

std::optional<cv::Vec3f> refine_circle_from_appearance(
    const cv::Mat& hsv,
    const cv::Point2f& estimated_center,
    float estimated_radius,
    const launch_monitor::vision::BallDetectorConfig& config) {
  const int margin = static_cast<int>(std::ceil(estimated_radius * 1.25F));
  const cv::Rect frame_bounds{0, 0, hsv.cols, hsv.rows};
  const cv::Rect candidate_region{
      static_cast<int>(std::lround(estimated_center.x)) - margin,
      static_cast<int>(std::lround(estimated_center.y)) - margin,
      margin * 2,
      margin * 2,
  };
  const cv::Rect region = candidate_region & frame_bounds;
  if (region.empty()) {
    return std::nullopt;
  }

  cv::Mat appearance_mask;
  cv::inRange(hsv(region),
              cv::Scalar{0.0, config.min_saturation, config.min_brightness},
              cv::Scalar{180.0, config.max_saturation, 255.0},
              appearance_mask);
  cv::morphologyEx(appearance_mask,
                   appearance_mask,
                   cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{5, 5}));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(appearance_mask,
                   contours,
                   cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  std::optional<cv::Vec3f> refined_circle;
  double largest_area = 0.0;
  for (const auto& contour : contours) {
    cv::Point2f local_center;
    float local_radius = 0.0F;
    cv::minEnclosingCircle(contour, local_center, local_radius);
    const cv::Point2f center = local_center + cv::Point2f{
        static_cast<float>(region.x), static_cast<float>(region.y)};
    const float center_distance = cv::norm(center - estimated_center);
    if (center_distance > estimated_radius * 0.60F ||
        local_radius < estimated_radius * 0.35F ||
        local_radius > estimated_radius * 1.20F) {
      continue;
    }

    const double area = cv::contourArea(contour);
    if (area <= largest_area) {
      continue;
    }

    largest_area = area;
    refined_circle = cv::Vec3f{center.x, center.y, local_radius};
  }

  return refined_circle;
}

}  // namespace

namespace launch_monitor::vision {

BallDetector::BallDetector(BallDetectorConfig config) : config_(config) {}

std::optional<BallObservation> BallDetector::locate_stationary_ball(
    const cv::Mat& frame,
    std::int64_t frame_index,
    double timestamp_ms) const {
  if (frame.empty() || frame.channels() != 3) {
    return std::nullopt;
  }

  const cv::Rect search_region =
      normalized_region_to_pixels(config_.pre_shot_search_region, frame.size());
  if (search_region.empty()) {
    return std::nullopt;
  }

  cv::Mat gray;
  cv::cvtColor(frame(search_region), gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);

  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(gray,
                   circles,
                   cv::HOUGH_GRADIENT,
                   1.2,
                   config_.min_radius_px * 2.0F,
                   100.0,
                   24.0,
                   static_cast<int>(config_.min_radius_px),
                   static_cast<int>(config_.max_radius_px));
  if (circles.empty()) {
    return std::nullopt;
  }

  cv::Mat hsv;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

  std::optional<BallObservation> best_observation;
  float best_score = -std::numeric_limits<float>::infinity();

  for (const cv::Vec3f& circle : circles) {
    cv::Point2f center{circle[0] + search_region.x,
                       circle[1] + search_region.y};
    float radius = circle[2];

    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::circle(mask, center, static_cast<int>(radius * 0.65F), cv::Scalar{255}, -1);
    const cv::Scalar mean_hsv = cv::mean(hsv, mask);
    const float mean_saturation = static_cast<float>(mean_hsv[1]);
    const float mean_brightness = static_cast<float>(mean_hsv[2]);

    if (mean_brightness < config_.min_brightness ||
        mean_saturation < config_.min_saturation ||
        mean_saturation > config_.max_saturation) {
      continue;
    }

    const float brightness_score = clamp_unit(
        (mean_brightness - config_.min_brightness) /
        (255.0F - config_.min_brightness));
    const float saturation_score = clamp_unit(
        (mean_saturation - config_.min_saturation) /
        (config_.max_saturation - config_.min_saturation));
    const float expected_radius =
        (config_.min_radius_px + config_.max_radius_px) * 0.5F;
    const float radius_score = clamp_unit(
        1.0F - std::abs(radius - expected_radius) / expected_radius);
    const float score = 0.55F * brightness_score + 0.30F * saturation_score +
                        0.15F * radius_score;

    if (score <= best_score) {
      continue;
    }

    if (const auto refined_circle =
            refine_circle_from_appearance(hsv, center, radius, config_);
        refined_circle.has_value()) {
      center = {(*refined_circle)[0], (*refined_circle)[1]};
      radius = (*refined_circle)[2];
    }

    best_score = score;
    best_observation = BallObservation{
        .frame_index = frame_index,
        .timestamp_ms = timestamp_ms,
        .center = center,
        .radius_px = radius,
        .confidence = score,
    };
  }

  return best_observation;
}

std::optional<BallObservation> BallDetector::locate_moving_ball(
    const cv::Mat& frame,
    const cv::Mat& motion_mask,
    const BallObservation& tee_ball,
    std::int64_t frame_index,
    double timestamp_ms) const {
  if (frame.empty() || frame.channels() != 3 || motion_mask.empty() ||
      motion_mask.type() != CV_8UC1 || frame.size() != motion_mask.size()) {
    return std::nullopt;
  }

  const cv::Mat corridor_mask = flight_corridor_mask(frame.size(), tee_ball, config_);
  if (corridor_mask.empty()) {
    return std::nullopt;
  }

  cv::Mat filtered_motion;
  cv::bitwise_and(motion_mask, corridor_mask, filtered_motion);
  cv::morphologyEx(filtered_motion,
                   filtered_motion,
                   cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{3, 3}));
  cv::morphologyEx(filtered_motion,
                   filtered_motion,
                   cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{7, 7}));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(filtered_motion,
                   contours,
                   cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return std::nullopt;
  }

  cv::Mat hsv;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  cv::Point2f direction = config_.expected_flight_direction;
  direction *= 1.0F / cv::norm(direction);

  std::optional<BallObservation> best_observation;
  float best_score = -std::numeric_limits<float>::infinity();
  for (const auto& contour : contours) {
    const float area = static_cast<float>(cv::contourArea(contour));
    if (area < config_.min_candidate_area_px ||
        area > config_.max_candidate_area_px) {
      continue;
    }

    const cv::Rect bounds = cv::boundingRect(contour);
    const float aspect_ratio = static_cast<float>(bounds.width) / bounds.height;
    if (aspect_ratio < config_.min_candidate_aspect_ratio ||
        aspect_ratio > config_.max_candidate_aspect_ratio) {
      continue;
    }

    cv::Point2f center;
    float radius = 0.0F;
    cv::minEnclosingCircle(contour, center, radius);
    if (radius < tee_ball.radius_px * config_.min_candidate_radius_tee_ratio ||
        radius > tee_ball.radius_px * config_.max_candidate_radius_tee_ratio) {
      continue;
    }

    const cv::Point2f from_tee = center - tee_ball.center;
    const float distance_from_tee = cv::norm(from_tee);
    const float forward_distance = from_tee.dot(direction);
    if (distance_from_tee < tee_ball.radius_px * config_.min_motion_distance_radii ||
        forward_distance <= 0.0F) {
      continue;
    }

    cv::Mat contour_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::drawContours(contour_mask,
                     std::vector<std::vector<cv::Point>>{contour},
                     -1,
                     cv::Scalar{255},
                     cv::FILLED);
    const cv::Scalar mean_hsv = cv::mean(hsv, contour_mask);
    const float mean_saturation = static_cast<float>(mean_hsv[1]);
    const float mean_brightness = static_cast<float>(mean_hsv[2]);
    if (mean_brightness < config_.min_brightness ||
        mean_saturation < config_.min_saturation ||
        mean_saturation > config_.max_saturation) {
      continue;
    }

    const float brightness_score = clamp_unit(
        (mean_brightness - config_.min_brightness) /
        (255.0F - config_.min_brightness));
    const float saturation_score = clamp_unit(
        (mean_saturation - config_.min_saturation) /
        (config_.max_saturation - config_.min_saturation));
    const float radius_score = clamp_unit(
        1.0F - std::abs(radius - tee_ball.radius_px) / tee_ball.radius_px);
    const float circularity = clamp_unit(
        static_cast<float>(4.0 * CV_PI * area /
                           std::pow(cv::arcLength(contour, true), 2.0)));
    const float direction_score = clamp_unit(forward_distance / distance_from_tee);
    const float score = 0.25F * brightness_score + 0.15F * saturation_score +
                        0.20F * radius_score + 0.25F * circularity +
                        0.15F * direction_score;
    if (score <= best_score) {
      continue;
    }

    best_score = score;
    best_observation = BallObservation{
        .frame_index = frame_index,
        .timestamp_ms = timestamp_ms,
        .center = center,
        .radius_px = radius,
        .confidence = score,
    };
  }

  if (!best_observation.has_value() ||
      best_observation->confidence < config_.min_candidate_confidence) {
    return std::nullopt;
  }

  return best_observation;
}

}  // namespace launch_monitor::vision
