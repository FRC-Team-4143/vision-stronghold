//Includes
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <limits.h>

#define SHOW

using namespace std;
using namespace cv;

// Main function
int main(int argc, char const *argv[]) {
    // Camera video stream
    VideoCapture camera;
    time_t start, end;
    unsigned int counter = 0;
    double sec;
    double fps;

    // Open Camera stream
    //camera.open("http://root:password@192.168.0.99/mjpg/video.mjpg"); // Used for IP
    camera.open(0); // Used for USB

    // Check if the camera is open
    if (!camera.isOpened()) {
        printf("Failed to open camera\n");
        return -1;
    }

#ifdef SHOW
    namedWindow("Contours", WINDOW_NORMAL); // Window to display Camera Stream
    namedWindow("Threshold", WINDOW_NORMAL); // Window to display the thresholded window
#endif
    
    Mat img; // Image from camera
    Mat hsv; // Image after translating to HSV
    Mat threshold; // Image after threshold
    Mat cannyO; // Image after canny edge detection
    Mat contourDrawing; // Image after Contours have been drawn on it

    vector<vector<Point> > contours; // Contours vector
    vector<Vec4i> hierarchy; // hierarchy of contours
    
    while (1) { 
        if (counter == 0) time(&start);

        camera >> img; // Grab Frame

        cvtColor(img, hsv, COLOR_BGR2HSV);  // convert to HSV
        //inRange(hsv, Scalar(0, 216, 220), Scalar(180, 255, 255), threshold); // Only take pixels within specified range
        inRange(hsv, Scalar(40, 0, 140), Scalar(255, 255, 255), threshold); // Only take pixels within specified range

        Canny(threshold, cannyO, 100, 200, 3); // Find edges
        findContours(cannyO, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0)); // Find contours
        
        contourDrawing = Mat::zeros(cannyO.size(), CV_8UC3); // Initialize Contour drawing

        for (int i = 0; i < contours.size(); i++){
            drawContours(img, contours, i, Scalar(255,i,255), 2, 8, hierarchy, 0, Point()); // Draw contours
            drawContours(contourDrawing, contours, i, Scalar(255,i,255), 2, 8, hierarchy, 0, Point()); // Draw contours
        }

#ifdef SHOW
        imshow("image", img); // Show camera view
        imshow("Threshold", threshold); // Show threshold view
        imshow("Contours", contourDrawing); // Show Contours
#endif

        //Check for escape to exit
        char k = (char)waitKey(10);
        if (k == 27) {
            break;
        }

        time(&end);
        counter++;
        sec = difftime(end,start);
        fps = counter/sec;
	if (counter%30==0) cout << "fps: " << fps << endl;
    }
    return 0;
}
