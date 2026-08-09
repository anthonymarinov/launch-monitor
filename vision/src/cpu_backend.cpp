#include <launch_monitor/vision/ball_detector.hpp>
#include <launch_monitor/vision/ball_tracker.hpp>
#include <launch_monitor/vision/main.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/geometry/2d.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void draw_tee_marker(cv::Mat& frame,
                     const launch_monitor::vision::BallObservation& tee_ball,
                     std::int64_t frame_index) {
  const cv::Point center{
      static_cast<int>(std::lround(tee_ball.center.x)),
      static_cast<int>(std::lround(tee_ball.center.y)),
  };
  if (frame_index == 0) {
    cv::circle(frame,
               center,
               static_cast<int>(std::lround(tee_ball.radius_px)),
               cv::Scalar(0, 165, 255),
               3);
    cv::putText(frame,
                "Tee ball",
                center + cv::Point{12, -12},
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 165, 255),
                2);
    return;
  }

  cv::drawMarker(frame,
                 center,
                 cv::Scalar(0, 165, 255),
                 cv::MARKER_CROSS,
                 18,
                 2);
}

void draw_track_overlay(cv::Mat& frame,
                        const launch_monitor::vision::BallTrack& track,
                        std::int64_t frame_index) {
  const cv::Scalar kTrackColor{255, 0, 255};
  const cv::Scalar kRecoveredColor{255, 255, 0};
  std::vector<cv::Point> visible_points;

  for (size_t index = 0; index < track.points.size(); ++index) {
    const auto& tracked_point = track.points[index];
    if (tracked_point.observation.frame_index > frame_index) {
      break;
    }

    const cv::Point center{
        static_cast<int>(std::lround(tracked_point.observation.center.x)),
        static_cast<int>(std::lround(tracked_point.observation.center.y)),
    };
    if (!visible_points.empty()) {
      cv::line(frame, visible_points.back(), center, kTrackColor, 2);
    }
    visible_points.push_back(center);
    const cv::Scalar& point_color = tracked_point.observation.recovered
                                        ? kRecoveredColor
                                        : kTrackColor;
    cv::circle(frame,
               center,
               static_cast<int>(std::lround(tracked_point.observation.radius_px)),
               point_color,
               3);

    if (tracked_point.observation.frame_index == frame_index) {
      cv::putText(frame,
                  (tracked_point.observation.recovered ? "Recovered " : "Track ") +
                      std::to_string(index + 1),
                  center + cv::Point{12, -12},
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.7,
                  point_color,
                  2);
    }
  }
}

}  // namespace

int launch_monitor::vision::cuda_device_count() {
  return 0;
}

bool launch_monitor::vision::frame_difference(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path) {
  cv::VideoCapture cap(input_path.string());
  if (!cap.isOpened()) {
    std::cerr << "error loading video file: " << input_path << '\n';
    return false;
  }

  cv::Mat frame, gray, background, diff, thresh;

  // read first frame to use as static background model
  cap >> frame;
  if (frame.empty()) {
    std::cerr << "video contains no readable frames: " << input_path << '\n';
    return false;
  }
  const cv::Size frame_size = frame.size();

  std::error_code error;
  std::filesystem::create_directories(output_path.parent_path(), error);
  if (error) {
    std::cerr << "error creating output directory: " << error.message() << '\n';
    return false;
  }

  const double input_fps = cap.get(cv::CAP_PROP_FPS);
  const double output_fps = input_fps > 0.0 ? input_fps : 30.0;

  const std::filesystem::path observations_path =
      output_path.parent_path() / "ball_observations.csv";
  std::ofstream observations_file(observations_path);
  if (!observations_file.is_open()) {
    std::cerr << "error creating observations file: " << observations_path << '\n';
    return false;
  }
  observations_file << "frame,timestamp_ms,x_px,y_px,radius_px,confidence,kind\n";
  observations_file << std::fixed << std::setprecision(3);

  const auto write_observation = [&observations_file](const BallObservation& observation,
                                                        const char* kind) {
    observations_file << observation.frame_index << ',' << observation.timestamp_ms << ','
                      << observation.center.x << ',' << observation.center.y << ','
                      << observation.radius_px << ',' << observation.confidence << ','
                      << kind << '\n';
  };

  cv::cvtColor(frame, background, cv::COLOR_BGR2GRAY);

  // apply slight blur to background to smooth camera sensor noise
  cv::GaussianBlur(background, background, cv::Size(5, 5), 0);

  BallDetector ball_detector;
  const auto tee_ball = ball_detector.locate_stationary_ball(
      frame,
      0,
      cap.get(cv::CAP_PROP_POS_MSEC));

  if (tee_ball.has_value()) {
    std::cout << "tee ball at (" << tee_ball->center.x << ", "
              << tee_ball->center.y << "), radius " << tee_ball->radius_px
              << " px, confidence " << tee_ball->confidence << '\n';
    write_observation(*tee_ball, "tee");
  } else {
    std::cerr << "could not locate a stationary tee ball in the first frame\n";
  }

  std::vector<BallObservation> candidate_observations;
  std::int64_t frame_index = 0;
  while (cap.read(frame)) {
    ++frame_index;
    const double timestamp_ms = cap.get(cv::CAP_PROP_POS_MSEC);

    // convert the current live frame to grayscale and blur it
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    // 1. Absolute Difference: Subtract the background from the current frame
    cv::absdiff(background, gray, diff);

    // 2. Thresholding: Turn any pixel that changed significantly into pure white (255)
    // Adjust the '30' threshold value depending on your video's lighting
    cv::threshold(diff, thresh, 30, 255, cv::THRESH_BINARY);

    if (tee_ball.has_value()) {
      const auto moving_ball = ball_detector.locate_moving_ball(
          frame, thresh, *tee_ball, frame_index, timestamp_ms);
      if (moving_ball.has_value()) {
        candidate_observations.push_back(*moving_ball);
        write_observation(*moving_ball, "candidate");
      }
    }
  }

  BallTracker ball_tracker;
  BallTrack track = ball_tracker.build_track(candidate_observations);
  if (const auto prediction = ball_tracker.predict_previous_observation(track);
      prediction.has_value()) {
    cv::VideoCapture recovery_cap(input_path.string());
    cv::Mat recovery_frame;
    for (std::int64_t index = 0;
         index <= prediction->frame_index && recovery_cap.read(recovery_frame);
         ++index) {
    }

    if (recovery_frame.empty()) {
      std::cerr << "could not read predicted recovery frame "
                << prediction->frame_index << '\n';
    } else {
      BallObservation recovery_prediction = *prediction;
      recovery_prediction.timestamp_ms = recovery_cap.get(cv::CAP_PROP_POS_MSEC);

      cv::Mat recovery_gray;
      cv::Mat recovery_diff;
      cv::Mat recovery_motion;
      cv::cvtColor(recovery_frame, recovery_gray, cv::COLOR_BGR2GRAY);
      cv::GaussianBlur(recovery_gray, recovery_gray, cv::Size(5, 5), 0);
      cv::absdiff(background, recovery_gray, recovery_diff);
      cv::threshold(recovery_diff, recovery_motion, 30, 255, cv::THRESH_BINARY);

      const auto recovered_ball = ball_detector.locate_near_prediction(
          recovery_frame, recovery_motion, recovery_prediction);
      if (recovered_ball.has_value()) {
        candidate_observations.push_back(*recovered_ball);
        write_observation(*recovered_ball, "recovered");
        track = ball_tracker.build_track(candidate_observations);
        std::cout << "recovered ball at frame " << recovered_ball->frame_index
                  << " with confidence " << recovered_ball->confidence << '\n';
      } else {
        std::cout << "no ball recovered at predicted frame "
                  << recovery_prediction.frame_index << '\n';
      }
    }
  }

  const std::filesystem::path track_path = output_path.parent_path() / "ball_track.csv";
  std::ofstream track_file(track_path);
  if (!track_file.is_open()) {
    std::cerr << "error creating track file: " << track_path << '\n';
    return false;
  }
  track_file << "frame,timestamp_ms,x_px,y_px,radius_px,confidence,"
                "vx_px_s,vy_px_s,speed_px_s,kind\n";
  track_file << std::fixed << std::setprecision(3);
  for (const TrackedBallObservation& point : track.points) {
    const BallObservation& observation = point.observation;
    track_file << observation.frame_index << ',' << observation.timestamp_ms << ','
               << observation.center.x << ',' << observation.center.y << ','
               << observation.radius_px << ',' << observation.confidence << ','
               << point.velocity_px_per_s.x << ',' << point.velocity_px_per_s.y << ','
               << point.speed_px_per_s << ','
               << (observation.recovered ? "recovered" : "detected") << '\n';
  }

  if (track.points.empty()) {
    std::cerr << "could not form a valid ball track from the candidates\n";
  } else {
    std::cout << "tracked " << track.points.size() << " ball observations; initial "
              << "velocity is (" << track.initial_velocity_px_per_s.x << ", "
              << track.initial_velocity_px_per_s.y << ") px/s\n";
  }

  cv::VideoCapture visualization_cap(input_path.string());
  if (!visualization_cap.isOpened()) {
    std::cerr << "error reopening video file for annotation: " << input_path << '\n';
    return false;
  }

  cv::VideoWriter output(
      output_path.string(),
      cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
      output_fps,
      frame_size);
  if (!output.isOpened()) {
    std::cerr << "error creating output video: " << output_path << '\n';
    return false;
  }

  frame_index = 0;
  while (visualization_cap.read(frame)) {
    if (tee_ball.has_value()) {
      draw_tee_marker(frame, *tee_ball, frame_index);
    }
    draw_track_overlay(frame, track, frame_index);
    output.write(frame);
    ++frame_index;
  }

  std::cout << "wrote tracking video to " << output_path << '\n';
  std::cout << "wrote ball observations to " << observations_path << '\n';
  std::cout << "wrote ball track to " << track_path << '\n';

  return true;
}
