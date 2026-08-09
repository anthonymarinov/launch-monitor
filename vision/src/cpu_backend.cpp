#include <launch_monitor/vision/ball_detector.hpp>
#include <launch_monitor/vision/main.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/geometry/2d.hpp>

#include <cmath>
#include <iostream>

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

  std::error_code error;
  std::filesystem::create_directories(output_path.parent_path(), error);
  if (error) {
    std::cerr << "error creating output directory: " << error.message() << '\n';
    return false;
  }

  const double input_fps = cap.get(cv::CAP_PROP_FPS);
  const double output_fps = input_fps > 0.0 ? input_fps : 30.0;
  cv::VideoWriter output(
      output_path.string(),
      cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
      output_fps,
      frame.size());
  if (!output.isOpened()) {
    std::cerr << "error creating output video: " << output_path << '\n';
    return false;
  }

  cv::cvtColor(frame, background, cv::COLOR_BGR2GRAY);

  // apply slight blur to background to smooth camera sensor noise
  cv::GaussianBlur(background, background, cv::Size(5, 5), 0);

  BallDetector ball_detector;
  const auto tee_ball = ball_detector.locate_stationary_ball(
      frame,
      0,
      cap.get(cv::CAP_PROP_POS_MSEC));

  cv::Mat annotated_first_frame = frame.clone();
  if (tee_ball.has_value()) {
    const cv::Point center{
        static_cast<int>(std::lround(tee_ball->center.x)),
        static_cast<int>(std::lround(tee_ball->center.y)),
    };
    cv::circle(annotated_first_frame,
               center,
               static_cast<int>(std::lround(tee_ball->radius_px)),
               cv::Scalar(0, 165, 255),
               3);
    cv::putText(annotated_first_frame,
                "Tee ball",
                center + cv::Point{12, -12},
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 165, 255),
                2);
    std::cout << "tee ball at (" << tee_ball->center.x << ", "
              << tee_ball->center.y << "), radius " << tee_ball->radius_px
              << " px, confidence " << tee_ball->confidence << '\n';
  } else {
    std::cerr << "could not locate a stationary tee ball in the first frame\n";
  }
  output.write(annotated_first_frame);

  while (true) {
    cap >> frame;
    if (frame.empty()) break; // end of video

    if (tee_ball.has_value()) {
      const cv::Point tee_center{
          static_cast<int>(std::lround(tee_ball->center.x)),
          static_cast<int>(std::lround(tee_ball->center.y)),
      };
      cv::drawMarker(frame,
                     tee_center,
                     cv::Scalar(0, 165, 255),
                     cv::MARKER_CROSS,
                     18,
                     2);
    }

    // convert the current live frame to grayscale and blur it
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    // 1. Absolute Difference: Subtract the background from the current frame
    cv::absdiff(background, gray, diff);

    // 2. Thresholding: Turn any pixel that changed significantly into pure white (255)
    // Adjust the '30' threshold value depending on your video's lighting
    cv::threshold(diff, thresh, 30, 255, cv::THRESH_BINARY);

    // 3. Find Contours: Group the white pixels into shapes
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw bounding boxes around anything moving
    for (size_t i = 0; i < contours.size(); i++) {
        // Filter out tiny specks of noise
        if (cv::contourArea(contours[i]) > 100) {
            cv::Rect bounding_box = cv::boundingRect(contours[i]);
            cv::rectangle(frame, bounding_box, cv::Scalar(0, 255, 0), 2);
        }
    }

    output.write(frame);
  }

  std::cout << "wrote tracking video to " << output_path << '\n';

  return true;
}
