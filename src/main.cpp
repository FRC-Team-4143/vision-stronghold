//Includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <limits.h>

#include "kernel.hpp"

//#define SHOW
//#define SAVEFILE

#define CAMSIZE 640,480 // lifecam
//#define IPADDRESS "10.41.43.255"
#define IPADDRESS "10.255.255.255"
#define PORT 4143
#define GREENDIST 30

static const float HEIGHT = 12.f; // inches
static const float HEIGHT_MM = 304.8f; // mm
static const float WIDTH = 20.f; // inches
static const float WIDTH_MM = 508.f; // mm
static const float RATIO = WIDTH / HEIGHT;
static const float RATIO_THRESHOLD = 2.0f * RATIO;
// 2016 static const float AREA_THRESHOLD = 3500; // px^2
static const float AREA_THRESHOLD = 1000; // px^2
//static const float AREA_THRESHOLD_TOP = 8000; // px^2
static const float AREA_THRESHOLD_TOP = 16000; // px^2
static const float HORIZ_VIEW_ANGLE_DEG = 60.f; //degrees // lifecam
static const float HORIZ_VIEW_ANGLE = HORIZ_VIEW_ANGLE_DEG * M_PI / 180.f; //radians
static const float VERT_VIEW_ANGLE_DEG = 60.f; //degrees // lifecam
static const float VERT_VIEW_ANGLE = VERT_VIEW_ANGLE_DEG * M_PI / 180.f; //radians

//2'8" x 4"
static const float TEST_H = 812.8f;
static const float TEST_W = 101.6f;

using namespace std;


#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#define WSAGetLastError() (errno)
#define SOCKET_ERROR (-1)

struct sockaddr_in si_other;
int s, slen;

void sendcenter(int s, struct sockaddr_in *si_other, int slen, cv::Rect best_fit, cv::Rect best_fit2) {
     int centerx = best_fit.x + best_fit.width/2 ;
     int centery = best_fit.y + best_fit.height/2;
     int center2x = best_fit2.x + best_fit2.width/2 ;
     int center2y = best_fit2.y + best_fit2.height/2;
      std::stringstream message;
      if (centerx != 0)
	centerx = centerx - 320;
      if (centery != 0)
	centery = centery - 240;
      if (center2x != 0)
	center2x = center2x - 320;
      if (center2y != 0)
	center2y = center2y - 240;
      message << std::to_string(centerx) << " " << std::to_string(centery) << 
         " " << std::to_string(center2x) << " " << std::to_string(center2y) << 
#ifdef GEAR
    " 1";
#elif GEAR2
    " 2";
#else
    " 0";
#endif
      std::cout << "sending " << message.str() << std::endl;
      if (sendto(s, message.str().c_str(), message.str().size(), 0, (struct sockaddr *) si_other, slen) == SOCKET_ERROR)
      {
        std::cout << "sendto() failed with error code : " << WSAGetLastError() << endl;
        //exit(EXIT_FAILURE);
      }
}

cv::Rect findRect(cv::Mat& hsv, cv::Mat& img)
{
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(hsv, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  //float min_ratio = 1000.f;
  //int min_dist_from_center = 1000;
  float best_area = 0;
  float best_area2 = 0;
  float area;
  cv::Rect best_fit;
  cv::Rect best_fit2;
  cv::Rect rect;
  bool found = false;
  for (auto& contour : contours) {
    rect = cv::boundingRect(contour);
#ifdef SHOW
    cv::rectangle(img, rect, cv::Scalar(0,255,0));
#endif

    area = rect.area();


    if ( area > best_area && area > AREA_THRESHOLD) {
       best_fit2 = best_fit;
       best_area2 = best_area;
   
       best_fit = rect;
       best_area = area;
       found = true;
       cout << " area "  << area;
    } else if ( area > best_area2  && area > AREA_THRESHOLD) {
       best_fit2 = rect;
       best_area2 = area;
    }
    

    //if (rect.area() > AREA_THRESHOLD && rect.area() < AREA_THRESHOLD_TOP) {
    //float ratio = fabs(float(rect.width) / float(rect.height) - RATIO);
    //cout << "ratio : " << ratio ;
    //if (ratio < RATIO_THRESHOLD /*&& ratio < min_ratio*/) {
    //  int centerx = rect.x + rect.width/2;
    //  if ( centerx < min_dist_from_center ) {
    //     best_area = rect.area();
    //     min_ratio = ratio;
    //     min_dist_from_center = centerx;
    //     best_fit = rect;
    //     found = true;
    //  }

/*
      if (ratio < RATIO_THRESHOLD && ratio < min_ratio) {
        best_area = rect.area();
        min_ratio = ratio;
        best_fit = rect;
        found = true;
      }
*/
  //  }
  }
  if (found) {
#ifdef SHOW
    cv::rectangle(img, best_fit, cv::Scalar(0,0,255));
    cv::rectangle(img, best_fit2, cv::Scalar(255,0,0));
#endif
    int centerx = best_fit.x + best_fit.width/2;
    int centery = best_fit.y + best_fit.height/2;
    sendcenter(s, &si_other, slen, best_fit, best_fit2);
    //cout << "found box x: " << centerx << " y: " << centery << " ratio: " << min_ratio << " area: " << best_area << endl;
    return best_fit;
  }
  //return rect;
  sendcenter(s, &si_other, slen, cv::Rect(), cv::Rect());
  return cv::Rect();
  //throw std::runtime_error("No rect found.");
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

  //struct sockaddr_in si_other;
  s = sizeof(si_other);
  slen = sizeof(si_other);

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
  inet_aton(IPADDRESS, &si_other.sin_addr);

    // Open Camera stream
    //camera.open("http://root:password@192.168.0.99/mjpg/video.mjpg"); // Used for IP
#ifdef GEAR
    camera.open(1); // Used for USB
#elif GEAR2
    camera.open(2); // Used for USB
#else
    camera.open(0); // Used for USB
#endif
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

    vector<vector<cv::Point> > contours; // Contours vector

   cv::cuda::GpuMat  gpuimage(cv::Size(CAMSIZE), CV_32FC4);
   cv::cuda::GpuMat  gpuhsv(cv::Size(CAMSIZE), CV_8UC4);
   cv::cuda::GpuMat  gputhresh(cv::Size(CAMSIZE), CV_8UC1);
   cv::cuda::GpuMat  gpuedges(cv::Size(CAMSIZE), CV_8UC1);
   cv::Mat img; 
   cv::Mat threshold; // Image after threshold
   cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edg = cv::cuda::createCannyEdgeDetector(100.0, 200.0, 3, false);

    camera >> img; // Grab Frame

#ifdef SAVEFILE
    const char* outFile = "./output.mjpeg";
    cv::VideoWriter outStream(outFile, CV_FOURCC('M','J','P','G'), 2, cv::Size(CAMSIZE), true );
#endif
    
    int count = 0;
    while (1) { 
        if (counter == 0) time(&start);

        gpuimage.upload(img);
	cv::cuda::cvtColor(gpuimage, gpuhsv, CV_RGB2HSV, 4);
	cuInRange(gpuhsv, gputhresh, 60-GREENDIST, 50, 50, 60+GREENDIST, 255, 255);
        gputhresh.download(threshold);


        canny_edg->detect(gputhresh, gpuedges);
        //cv::cuda::Canny(gputhresh, gpuedges, 100, 200, 3);
        cv::Mat cpuedges(gpuedges);
        cv::Rect leftc = findRect(cpuedges, img);


#ifdef SAVEFILE
        // MJPEG BEGIN
        outStream.write(img);
        // MJPEG END
#endif

        camera >> img; // Grab Frame

#ifdef SHOW
        if(counter % 30 == 0) {
		cv::imshow("image", img); // Show camera view
		cv::imshow("canny", cpuedges);
		cv::imshow("Threshold", threshold); // Show threshold view
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
