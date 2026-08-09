#include <launch_monitor/vision/main.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/geometry/2d.hpp>

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

  while (true) {
    cap >> frame;
    if (frame.empty()) break; // end of video

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
