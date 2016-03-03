//standard include
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <iomanip>

#define ZED
#ifdef ZED
//ZED include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>
using namespace sl::zed;
#endif

//OpenCV include
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

// Cuda functions include
#include "kernel.cuh"


#define IPADDRESS "10.41.43.60"
#define PORT 4143

static const float HEIGHT = 12.f; // inches
static const float HEIGHT_MM = 304.8f; // mm
static const float WIDTH = 20.f; // inches
static const float WIDTH_MM = 508.f; // mm
static const float RATIO = WIDTH / HEIGHT;
static const float RATIO_THRESHOLD = 0.2f * RATIO;
static const float AREA_THRESHOLD = 3000; // px^2
static const float AREA_THRESHOLD_TOP = 8000; // px^2
static const float HORIZ_VIEW_ANGLE_DEG = 110.f; //degrees
static const float HORIZ_VIEW_ANGLE = HORIZ_VIEW_ANGLE_DEG * M_PI / 180.f; //radians
static const float VERT_VIEW_ANGLE_DEG = 110.f; //degrees
static const float VERT_VIEW_ANGLE = VERT_VIEW_ANGLE_DEG * M_PI / 180.f; //radians

//2'8" x 4"
static const float TEST_H = 812.8f;
static const float TEST_W = 101.6f;

#ifdef __linux__
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#define WSAGetLastError() (errno)
#define SOCKET_ERROR (-1)
#else
#pragma comment(lib,"ws2_32.lib") //Winsock Library
#endif


float calculateDistance(float real_dim, float measured_px, float focal_length_px)
{
  //return real_dim * total_px / (2.f * measured_px * std::tanf(view_angle / 2.f));
  return real_dim / measured_px * focal_length_px;
}

cv::Rect findRect(cv::Mat& hsv)
{
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(hsv, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  float min_ratio = 1000.f;
  cv::Rect best_fit;
  bool found = false;
  for (auto& contour : contours) {
    cv::Rect rect = cv::boundingRect(contour);
    if (rect.area() > AREA_THRESHOLD && rect.area() < AREA_THRESHOLD_TOP) {
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

cv::Rect findRect(const Mat& d_mat, cv::gpu::GpuMat& d_hsv, cv::gpu::GpuMat& d_thresh)
{
  cv::Mat hsv;
  cv::gpu::GpuMat d_left_cv(d_mat.height, d_mat.width, d_mat.data_type == sl::zed::UCHAR ? CV_8UC4 : CV_32FC4, d_mat.data, d_mat.step);
  cv::gpu::cvtColor(d_left_cv, d_hsv, CV_RGB2HSV, 4);
  cuInRange(d_hsv, d_thresh, 30, 0, 160, 65, 255, 255);
  d_thresh.download(hsv);
  //for (int value = 255; value > 0; value -= 10) {
  //  std::cout << value << std::endl;
  //  cuInRange(d_hsv, d_thresh, 30, 0, 160, 65, 255, 255);
  //  d_thresh.download(hsv);
  //  cv::imshow("hsv", hsv);
  //  cv::waitKey(2000);
  //}
  return findRect(hsv);
}

cv::Rect findRect_cpu(cv::Mat& mat)
{
  cv::Mat hsv;
  cv::cvtColor(mat, hsv, CV_RGB2HSV);
  cv::inRange(hsv, cv::Scalar(40, 0, 140), cv::Scalar(255, 255, 255), hsv);
  return findRect(hsv);
}

#ifdef ZED
float findDepthMaxConfidence(Camera* cam, const cv::Rect& rect)
{
  Mat confidence = cam->retrieveMeasure(CONFIDENCE);
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
  Mat depth = cam->retrieveMeasure(DEPTH);
  float* depth_ptr = (float*)depth.data;
  return depth_ptr[min_conf.y * (depth.step/sizeof(float))+min_conf.y];
}
#endif

void Send(int s, std::string message, struct sockaddr_in& si_other, int slen)
{
#define BUFLEN 512
  char buf[BUFLEN];
  //send the message
  std::cout << "Sending " << message.c_str() << std::endl;
  if (sendto(s, message.c_str(), message.size(), 0, (struct sockaddr *) &si_other, slen) == SOCKET_ERROR)
  {
    printf("sendto() failed with error code : %d", WSAGetLastError());
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv) {

  if (argc > 2) {
    std::cout << "Only the path of an image can be passed in arg" << std::endl;
    return -1;
  }

  Camera *cam;
  if (argc == 1) {
    cam = new Camera(HD720);
    cam->setCameraSettingsValue(ZED_BRIGHTNESS, 0);
    cam->setCameraSettingsValue(ZED_WHITEBALANCE, 3500);
  }
  else {
    cam = new Camera(argv[1]);
  }

  ERRCODE err = cam->init(MODE::QUALITY, -1, false, false, false);
  std::cout << errcode2str(err) << std::endl;
  if (err != SUCCESS) {
    delete cam;
    return 1;
  }

  /*
  Algorithm
  =========
  * Grab images from cam with depth calculation
  * Convert left image to HSV (on gpu)
  * Threshold HSV image (on gpu)
  * Open/close HSV image? (on gpu)
  * Download binary result image from gpu
  * FindContours
  */

  // get the focale and the baseline of the cam
  float fx = cam->getParameters()->LeftCam.fx;
  float fy = cam->getParameters()->LeftCam.fy;

#ifndef __linux__
  WSADATA wsa;
  //Initialise winsock
  printf("\nInitialising Winsock...");
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
  {
    printf("Failed. Error Code : %d", WSAGetLastError());
    exit(EXIT_FAILURE);
  }
  printf("Initialised.\n");
#endif
  struct sockaddr_in si_other;
  int s, slen = sizeof(si_other);

  //create socket
  if ((s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == SOCKET_ERROR) {
    printf("socket() failed with error code : %d", WSAGetLastError());
    exit(EXIT_FAILURE);
  }

  //setup address structure
  memset((char *)&si_other, 0, sizeof(si_other));
  si_other.sin_family = AF_INET;
  si_other.sin_port = htons(PORT);
#ifndef __linux__
  si_other.sin_addr.S_un.S_addr = inet_addr(IPADDRESS);
#else
  inet_aton(IPADDRESS, &si_other.sin_addr);
#endif

  int total_width = cam->getImageSize().width;
  int total_height = cam->getImageSize().height;

  char key = ' ';
  bool run = true;
  bool calibrate = true;
  cv::gpu::GpuMat d_hsv(cam->getImageSize().height, cam->getImageSize().width, CV_8UC4);
  cv::gpu::GpuMat d_thresh(cam->getImageSize().height, cam->getImageSize().width, CV_8UC1);
  while (run) {

    // Grab the current images and compute the disparity
    bool res = cam->grab(RAW);
    try {
      cv::Rect left = findRect(cam->retrieveImage_gpu(LEFT), d_hsv, d_thresh);
      //cv::Rect right = findRect(cam->retrieveImage_gpu(RIGHT), d_hsv, d_thresh);
      //Mat left_img = cam->retrieveImage(LEFT);
      //Mat right_img = cam->retrieveImage(RIGHT);
      //cv::Mat left_img_cv = slMat2cvMat(left_img);
      //cv::Mat right_img_cv = slMat2cvMat(right_img);
      //cv::Rect left = findRect_cpu(left_img_cv);
      //cv::Rect right = findRect_cpu(right_img_cv);

#ifdef ZED
      float distance_cam = findDepthMaxConfidence(cam, left);
#else
      float distance_cam = 0.;
#endif
      float distance_img = calculateDistance(WIDTH_MM, left.width, fx);
      float distance_img_h = calculateDistance(HEIGHT_MM, left.height, fy);

      float yaw = (left.x + left.width / 2. - total_width / 2.) / total_width * 2.;
      float pitch = (total_height / 2. - left.y + left.height / 2.) / total_height * 2.;
      std::stringstream message;
      message << std::to_string(distance_img_h) << "," << std::to_string(yaw) << "," << std::to_string(pitch);
      std::cout << message.str() << std::endl;
      if (sendto(s, message.str().c_str(), message.str().size(), 0, (struct sockaddr *) &si_other, slen) == SOCKET_ERROR)
      {
        printf("sendto() failed with error code : %d", WSAGetLastError());
        exit(EXIT_FAILURE);
      }


      if (calibrate) {
        std::cout << std::setprecision(0) << std::fixed << distance_cam << ", " << distance_img << ", " << distance_img_h << std::endl;
        cv::Mat left_cv = slMat2cvMat(cam->retrieveImage(LEFT));
        cv::Point center;
        center.x = left.x + left.width / 2.;
        center.y = left.y + left.height / 2.;
        cv::circle(left_cv, center, 10, cv::Scalar(0, 0, 255, 1), 3);
        center.y += 50;
        cv::putText(left_cv, std::to_string(distance_cam), center,
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
    catch (...) {
    }
  }

#ifndef __linux__
  closesocket(s);
  WSACleanup();
#else
  close(s);
#endif
  delete cam;
  return 0;
}
