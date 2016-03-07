//Includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <limits.h>

#include "kernel.hpp"

#define SHOW
//#define SAVEFILE
#define FINDRECT

//#define CAMSIZE 1280,480
#define CAMSIZE 640,480
#define IPADDRESS "10.41.43.255"
#define PORT 4143

static const float HEIGHT = 12.f; // inches
static const float HEIGHT_MM = 304.8f; // mm
static const float WIDTH = 20.f; // inches
static const float WIDTH_MM = 508.f; // mm
static const float RATIO = WIDTH / HEIGHT;
static const float RATIO_THRESHOLD = 0.4f * RATIO;
static const float AREA_THRESHOLD = 3000; // px^2
//static const float AREA_THRESHOLD_TOP = 8000; // px^2
static const float AREA_THRESHOLD_TOP = 15000; // px^2
static const float HORIZ_VIEW_ANGLE_DEG = 110.f; //degrees
static const float HORIZ_VIEW_ANGLE = HORIZ_VIEW_ANGLE_DEG * M_PI / 180.f; //radians
static const float VERT_VIEW_ANGLE_DEG = 110.f; //degrees
static const float VERT_VIEW_ANGLE = VERT_VIEW_ANGLE_DEG * M_PI / 180.f; //radians

//2'8" x 4"
static const float TEST_H = 812.8f;
static const float TEST_W = 101.6f;


using namespace std;


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


cv::Rect findRect(cv::Mat& hsv)
{
  std::vector<std::vector<cv::Point>> contours;
  //cv::findContours(hsv, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  cv::findContours(hsv, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  float min_ratio = 1000.f;
  float best_area;
  cv::Rect best_fit;
  cv::Rect rect;
  bool found = false;
  for (auto& contour : contours) {
    rect = cv::boundingRect(contour);
    if (rect.area() > AREA_THRESHOLD && rect.area() < AREA_THRESHOLD_TOP) {
      float ratio = fabs(float(rect.width) / float(rect.height) - RATIO);
      if (ratio < RATIO_THRESHOLD && ratio < min_ratio) {
        best_area = rect.area();
        min_ratio = ratio;
        best_fit = rect;
        found = true;
      }
    }
  }
  if (found) {
    int centerx = best_fit.x + best_fit.width/2;
    int centery = best_fit.y + best_fit.height/2;
    cout << "found box x: " << centerx << " y: " << centery << " ratio: " << min_ratio << " area: " << best_area << endl;
    return best_fit;
  }
  //return rect;
  return cv::Rect();
  //throw std::runtime_error("No rect found.");
}

void sendcenter(int s, struct sockaddr_in *si_other, int slen, int center) {
      std::stringstream message;
      if (center != 0)
	center = center - 320;
      message << std::to_string(center);
      std::cout << "sending " << message.str() << std::endl;
      if (sendto(s, message.str().c_str(), message.str().size(), 0, (struct sockaddr *) si_other, slen) == SOCKET_ERROR)
      {
        std::cout << "sendto() failed with error code : " << WSAGetLastError() << endl;
        //exit(EXIT_FAILURE);
      }
}


// Main function
int main(int argc, char const *argv[]) {
    // Camera video stream
    cv::VideoCapture camera;
    time_t start, end;
    unsigned int counter = 0;
    double sec;
    double fps;

    cout << cv::getBuildInformation();


    // start networking

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
    std::cout << "sendto() failed with error code : " << WSAGetLastError() << endl;
    exit(EXIT_FAILURE);
  }
  int broadcastEnable = 1;
  int ret = setsockopt(s, SOL_SOCKET, SO_BROADCAST, &broadcastEnable, sizeof(broadcastEnable));
  cout << "broadcast ret: " << ret << endl;

  //setup address structure
  memset((char *)&si_other, 0, sizeof(si_other));
  si_other.sin_family = AF_INET;
  si_other.sin_port = htons(PORT);
#ifndef __linux__
  si_other.sin_addr.S_un.S_addr = inet_addr(IPADDRESS);
#else
  inet_aton(IPADDRESS, &si_other.sin_addr);
#endif


    // Open Camera stream
    //camera.open("http://root:password@192.168.0.99/mjpg/video.mjpg"); // Used for IP
    camera.open(0); // Used for USB
    //camera.set(CV_CAP_PROP_FPS,100); doesn't work

    // Check if the camera is open
    if (!camera.isOpened()) {
        printf("Failed to open camera\n");
        return -1;
    }

#ifdef SHOW
    cv::namedWindow("Threshold", cv::WINDOW_NORMAL); // Window to display the thresholded window
#endif
    
    cv::Mat hsv; // Image after translating to HSV
    //cv::Mat cannyO; // Image after canny edge detection
    //cv::Mat contourDrawing; // Image after Contours have been drawn on it

    vector<vector<cv::Point> > contours; // Contours vector
    vector<cv::Vec4i> hierarchy; // hierarchy of contours

   cv::gpu::CudaMem page_locked(cv::Size(CAMSIZE), CV_32FC3);
   cv::gpu::CudaMem threshpage_locked(cv::Size(CAMSIZE), CV_8UC1);
   cv::gpu::GpuMat  gpuimage(cv::Size(CAMSIZE), CV_32FC3);
   cv::gpu::GpuMat  gpuhsv(cv::Size(CAMSIZE), CV_8UC3);
   cv::gpu::GpuMat  gputhresh(cv::Size(CAMSIZE), CV_8UC1);
   cv::Mat img = page_locked;
   cv::Mat threshold = threshpage_locked; // Image after threshold


   cv::gpu::CudaMem page_locked2(cv::Size(CAMSIZE), CV_32FC3);
   cv::gpu::CudaMem threshpage_locked2(cv::Size(CAMSIZE), CV_8UC1);
   cv::gpu::GpuMat  gpuimage2(cv::Size(CAMSIZE), CV_32FC3);
   cv::gpu::GpuMat  gpuhsv2(cv::Size(CAMSIZE), CV_8UC3);
   cv::gpu::GpuMat  gputhresh2(cv::Size(CAMSIZE), CV_8UC1);
   cv::Mat img2 = page_locked2;
   cv::Mat threshold2 = threshpage_locked2; // Image after threshold

   cv::gpu::Stream stream;

    camera >> img; // Grab Frame

    const char* outFile = "./output.mjpeg";
    cv::VideoWriter outStream(outFile, CV_FOURCC('M','J','P','G'), 2, cv::Size(CAMSIZE), true );
    
    while (1) { 
        if (counter == 0) time(&start);

	stream.enqueueUpload(img, gpuimage);
	cv::gpu::cvtColor(gpuimage, gpuhsv, CV_RGB2HSV, 3, stream);
	cuInRange(gpuhsv, gputhresh, 60-30, 90, 90, 60+30, 255, 255, stream);
	stream.enqueueDownload(gputhresh, threshold);
#ifdef FINDRECT
        cv::Rect left = findRect(threshold2);
        sendcenter(s, &si_other, slen, left.x + left.width/2);
	cv::rectangle(threshold2, left, cv::Scalar(255));
#endif
#ifdef SHOW
        cv::imshow("Threshold", threshold2); // Show threshold view
        cv::imshow("image", img); // Show camera view
#endif
        camera >> img2; // Grab Frame
	stream.waitForCompletion();

	stream.enqueueUpload(img2, gpuimage2);
	cv::gpu::cvtColor(gpuimage2, gpuhsv2, CV_RGB2HSV, 3, stream);
	cuInRange(gpuhsv2, gputhresh2, 60-30, 90, 90, 60+30, 255, 255, stream);
	stream.enqueueDownload(gputhresh2, threshold2);

        //cvtColor(img, hsv, COLOR_BGR2HSV);  // convert to HSV
        //inRange(hsv, Scalar(0, 216, 220), Scalar(180, 255, 255), threshold); // Only take pixels within specified range
        //inRange(hsv, Scalar(40, 0, 140), Scalar(255, 255, 255), threshold); // Only take pixels within specified range

#ifdef FINDRECT
        cv::Rect right = findRect(threshold);
        sendcenter(s, &si_other, slen, right.x + right.width/2);
	cv::rectangle(threshold, right, cv::Scalar(255));
#endif
#ifdef SHOW
        cv::imshow("Threshold", threshold); // Show threshold view
        cv::imshow("image", img2); // Show camera view
#endif

#ifdef SAVEFILE
        // MJPEG BEGIN
        outStream.write(img);
        // MJPEG END
#endif

        camera >> img; // Grab Frame
	stream.waitForCompletion();

        //Check for escape to exit
        char k = (char)cv::waitKey(10);
        if (k == 27) {
            break;
        }

        time(&end);
        counter++;
        counter++;
        sec = difftime(end,start);
        fps = counter/sec;
	if (counter%30==0) cout << "fps: " << fps << endl;
    }
    return 0;
}
