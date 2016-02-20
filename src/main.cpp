//standard include
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <chrono>

//ZED include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

//OpenCV include
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Cuda functions include
#include "kernel.cuh"

using namespace sl::zed;

static const float HEIGHT = 12.f; // inches
static const float HEIGHT_MM = 304.8f; // mm
static const float WIDTH = 20.f; // inches
static const float WIDTH_MM = 508.f; // mm
static const float RATIO = WIDTH / HEIGHT;
static const float RATIO_THRESHOLD = 0.2f * RATIO;
static const float AREA_THRESHOLD = 3000; // px^2
static const float HORIZ_VIEW_ANGLE_DEG = 110.f; //degrees
static const float HORIZ_VIEW_ANGLE = HORIZ_VIEW_ANGLE_DEG * M_PI / 180.f; //radians
static const float VERT_VIEW_ANGLE_DEG = 110.f; //degrees
static const float VERT_VIEW_ANGLE = VERT_VIEW_ANGLE_DEG * M_PI / 180.f; //radians

//2'8" x 4"
static const float TEST_H = 812.8f;
static const float TEST_W = 101.6f;


float calculateDistance(float real_dim, float measured_px, float focal_length_px)
{
  //return real_dim * total_px / (2.f * measured_px * std::tanf(view_angle / 2.f));
  return real_dim / measured_px * focal_length_px;
}

cv::Rect findRect(Mat& d_mat, cv::gpu::GpuMat& d_hsv)
{
  cv::Mat hsv;
  cv::gpu::GpuMat d_left_cv(d_mat.height, d_mat.width, d_mat.data_type == sl::zed::UCHAR ? CV_8UC4 : CV_32FC4, d_mat.data, d_mat.step);
  cuInRange(d_left_cv, d_hsv, 0, 0, 80, 255, 255, 255);
  d_hsv.download(hsv);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(hsv, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  float min_ratio = FLT_MAX;
  cv::Rect best_fit;
  bool found = false;
  for (auto& contour : contours) {
    cv::Rect rect = cv::boundingRect(contour);
    if (rect.area() > AREA_THRESHOLD) {
      float ratio = fabs(float(rect.width) / float(rect.height) - RATIO);
      if (ratio < RATIO_THRESHOLD && ratio < min_ratio) {
        min_ratio = ratio;
        best_fit = rect;
        found = true;
      }
    }
  }
  if (found) {
    return best_fit;
  }
  throw std::runtime_error("No rect found.");
}

float findDepthMaxConfidence(Camera* zed, const cv::Rect& rect)
{
  Mat confidence = zed->retrieveMeasure(CONFIDENCE);
  float* conf_ptr = (float*)confidence.data;
  cv::Point min_conf;
  float min = 100.f;
  for (int x = rect.x; x < rect.x + rect.width; x++) {
    for (int y = rect.y; y < rect.y + rect.height; y++) {
      if (conf_ptr[y * (confidence.step/sizeof(float)) + x] < min) {
        min = conf_ptr[y * (confidence.step/sizeof(float)) + x];
        min_conf.x = x;
        min_conf.y = y;
      }
    }
  }
  Mat depth = zed->retrieveMeasure(DEPTH);
  float* depth_ptr = (float*)depth.data;
  return depth_ptr[min_conf.y * (depth.step/sizeof(float))+min_conf.y];
}

int main(int argc, char **argv) {

  if (argc > 2) {
    std::cout << "Only the path of an image can be passed in arg" << std::endl;
    return -1;
  }

  Camera *zed;
  if (argc == 1) {
    zed = new Camera(HD720);
    zed->setCameraSettingsValue(ZED_BRIGHTNESS, 0);
    zed->setCameraSettingsValue(ZED_WHITEBALANCE, 3500);
  }
  else {
    zed = new Camera(argv[1]);
  }

  ERRCODE err = zed->init(MODE::QUALITY, -1, false, false, false);
  std::cout << errcode2str(err) << std::endl;
  if (err != SUCCESS) {
    delete zed;
    return 1;
  }

  /*
  Algorithm
  =========
  * Grab images from zed with depth calculation
  * Convert left image to HSV (on gpu)
  * Threshold HSV image (on gpu)
  * Open/close HSV image? (on gpu)
  * Download binary result image from gpu
  * FindContours
  */

  // get the focale and the baseline of the zed
  float fx = zed->getParameters()->LeftCam.fx;
  float fy = zed->getParameters()->LeftCam.fy;

  char key = ' ';
  bool run = true;
  bool calibrate = true;
  cv::gpu::GpuMat d_hsv(zed->getImageSize().height, zed->getImageSize().width, CV_8UC1);
  while (run) {

    // Grab the current images and compute the disparity
    bool res = zed->grab(RAW);
    cv::Rect left = findRect(zed->retrieveImage_gpu(LEFT), d_hsv);
    cv::Rect right = findRect(zed->retrieveImage_gpu(RIGHT), d_hsv);

    float distance_zed = findDepthMaxConfidence(zed, left);
    float distance_img = calculateDistance(WIDTH_MM, left.width, fx);
    float distance_img_h = calculateDistance(HEIGHT_MM, left.height, fy);

    if (calibrate) {
      cv::Mat left_cv = slMat2cvMat(zed->retrieveImage(LEFT));
      cv::Point center;
      center.x = left.x + left.width / 2.;
      center.y = left.y + left.height / 2.;
      cv::circle(left_cv, center, 10, cv::Scalar(0, 0, 255, 1), 3);
      center.y += 50;
      cv::putText(left_cv, std::to_string(distance_zed), center,
                  CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200, 255),
                  2);
      center.y -= 15;
      cv::putText(left_cv, std::to_string(distance_img), center,
                  CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200, 255),
                  2);
      center.y -= 15;
      cv::putText(left_cv, std::to_string(distance_img_h), center,
                  CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200, 255),
                  2);
      cv::imshow("Test", left_cv);
      key = cv::waitKey(2000);

      switch (key) // handle the pressed key
      {
      case 'q': // close the program
      case 'Q':
        run = false;
        break;
      default:
        break;
      }
    }
  }

  delete zed;
  return 0;
}
