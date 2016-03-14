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

//#define SHOW
//#define SAVEFILE

//#define CAMSIZE 1280,480 // zed
#define CAMSIZE 640,480 // lifecam
#define IPADDRESS "10.41.43.255"
#define PORT 4143
#define GREENDIST 30

static const float HEIGHT = 12.f; // inches
static const float HEIGHT_MM = 304.8f; // mm
static const float WIDTH = 20.f; // inches
static const float WIDTH_MM = 508.f; // mm
static const float RATIO = WIDTH / HEIGHT;
static const float RATIO_THRESHOLD = 2.0f * RATIO;
static const float AREA_THRESHOLD = 3500; // px^2
//static const float AREA_THRESHOLD_TOP = 8000; // px^2
static const float AREA_THRESHOLD_TOP = 10000; // px^2
//static const float HORIZ_VIEW_ANGLE_DEG = 110.f; //degrees // zed
static const float HORIZ_VIEW_ANGLE_DEG = 60.f; //degrees // lifecam
static const float HORIZ_VIEW_ANGLE = HORIZ_VIEW_ANGLE_DEG * M_PI / 180.f; //radians
//static const float VERT_VIEW_ANGLE_DEG = 110.f; //degrees // zed
static const float VERT_VIEW_ANGLE_DEG = 60.f; //degrees // lifecam
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


cv::Rect findRect(cv::Mat& hsv, cv::Mat& img)
{
  std::vector<std::vector<cv::Point>> contours;
  //cv::findContours(hsv, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  cv::findContours(hsv, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  float min_ratio = 1000.f;
  int min_dist_from_center = 1000;
  float best_area;
  cv::Rect best_fit;
  cv::Rect rect;
  bool found = false;
  for (auto& contour : contours) {
    rect = cv::boundingRect(contour);
#ifdef SHOW
    cv::rectangle(img, rect, cv::Scalar(0,255,0));
#endif

    if (rect.area() > AREA_THRESHOLD && rect.area() < AREA_THRESHOLD_TOP) {
    float ratio = fabs(float(rect.width) / float(rect.height) - RATIO);
    if (ratio < RATIO_THRESHOLD /*&& ratio < min_ratio*/) {
      int centerx = abs(best_fit.x + best_fit.width/2 - 320);
      if ( centerx < min_dist_from_center ) {
         best_area = rect.area();
         min_ratio = ratio;
         min_dist_from_center = centerx;
         best_fit = rect;
         found = true;
      }

/*
      if (ratio < RATIO_THRESHOLD && ratio < min_ratio) {
        best_area = rect.area();
        min_ratio = ratio;
        best_fit = rect;
        found = true;
      }
*/
    }
    }
  }
  if (found) {
#ifdef SHOW
    cv::rectangle(img, best_fit, cv::Scalar(0,0,255));
#endif
    int centerx = best_fit.x + best_fit.width/2;
    int centery = best_fit.y + best_fit.height/2;
    cout << "found box x: " << centerx << " y: " << centery << " ratio: " << min_ratio << " area: " << best_area << endl;
    return best_fit;
  }
  //return rect;
  return cv::Rect();
  //throw std::runtime_error("No rect found.");
}

void sendcenter(int s, struct sockaddr_in *si_other, int slen, cv::Rect best_fit) {
     int centerx = best_fit.x + best_fit.width/2 ;
     int centery = best_fit.y + best_fit.height/2;
      std::stringstream message;
      if (centerx != 0)
	centerx = centerx - 320;
      if (centery != 0)
	centery = centery - 240;
      message << std::to_string(centerx) << " " << std::to_string(centery);
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

   cv::gpu::CudaMem page_locked(cv::Size(CAMSIZE), CV_32FC4);
   cv::gpu::CudaMem threshpage_locked(cv::Size(CAMSIZE), CV_8UC1);
   cv::gpu::GpuMat  gpuimage(cv::Size(CAMSIZE), CV_32FC4);
   cv::gpu::GpuMat  gpuhsv(cv::Size(CAMSIZE), CV_8UC4);
   cv::gpu::GpuMat  gputhresh(cv::Size(CAMSIZE), CV_8UC1);
   cv::gpu::GpuMat  gpuedges(cv::Size(CAMSIZE), CV_8UC1);
   cv::Mat img = page_locked;
   cv::Mat threshold = threshpage_locked; // Image after threshold

   cv::gpu::CudaMem page_locked2(cv::Size(CAMSIZE), CV_32FC4);
   cv::gpu::CudaMem threshpage_locked2(cv::Size(CAMSIZE), CV_8UC1);
   cv::gpu::GpuMat  gpuimage2(cv::Size(CAMSIZE), CV_32FC4);
   cv::gpu::GpuMat  gpuhsv2(cv::Size(CAMSIZE), CV_8UC4);
   cv::gpu::GpuMat  gputhresh2(cv::Size(CAMSIZE), CV_8UC1);
   cv::gpu::GpuMat  gpuedges2(cv::Size(CAMSIZE), CV_8UC1);
   cv::Mat img2 = page_locked2;
   cv::Mat threshold2 = threshpage_locked2; // Image after threshold

   cv::gpu::Stream stream;

    camera >> img; // Grab Frame

#ifdef SAVEFILE
    const char* outFile = "./output.mjpeg";
    cv::VideoWriter outStream(outFile, CV_FOURCC('M','J','P','G'), 2, cv::Size(CAMSIZE), true );
#endif
    
    int count = 0;
    while (1) { 
        if (counter == 0) time(&start);

	stream.enqueueUpload(img, gpuimage);
	cv::gpu::cvtColor(gpuimage, gpuhsv, CV_RGB2HSV, 4, stream);
	cuInRange(gpuhsv, gputhresh, 60-GREENDIST, 50, 50, 60+GREENDIST, 255, 255, stream);
	stream.enqueueDownload(gputhresh, threshold);

        camera >> img2; // Grab Frame
	stream.waitForCompletion();

        cv::gpu::Canny(gputhresh, gpuedges, 100, 200, 3);
        cv::Mat cpuedges(gpuedges);
        cv::Rect leftc = findRect(cpuedges, img);
        sendcenter(s, &si_other, slen, leftc);
#ifdef SHOW
        //cv::imshow("Threshold", threshold2); // Show threshold view
        //cv::imshow("image", img); // Show camera view
        //cv::imshow("canny", cpuedges);
#endif

	stream.enqueueUpload(img2, gpuimage2);
	cv::gpu::cvtColor(gpuimage2, gpuhsv2, CV_RGB2HSV, 4, stream);
	cuInRange(gpuhsv2, gputhresh2, 60-GREENDIST, 50, 50, 60+GREENDIST, 255, 255, stream);
	stream.enqueueDownload(gputhresh2, threshold2);

#ifdef SAVEFILE
        // MJPEG BEGIN
        outStream.write(img);
        // MJPEG END
#endif

        camera >> img; // Grab Frame
	stream.waitForCompletion();

        cv::gpu::Canny(gputhresh2, gpuedges2, 100, 200, 3);
        cv::Mat cpuedges2(gpuedges2);
        cv::Rect rightc = findRect(cpuedges2, img2);
        sendcenter(s, &si_other, slen, rightc);
#ifdef SHOW
        if(counter % 30 == 0) {
        cv::imshow("Threshold", threshold); // Show threshold view
        cv::imshow("image", img2); // Show camera view
        cv::imshow("canny", cpuedges2);
	count = 0;
	}
#endif
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
