//standard include
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
using namespace std;

//main Loop

int main(int argc, char **argv) {

  if (argc > 2) {
    std::cout << "Only the path of an image can be passed in arg" << std::endl;
    return -1;
  }

  Camera *zed;
  if (argc == 1) {
    zed = new Camera(HD720);
  }
  else {
    zed = new Camera(argv[1]);
  }

  // init computation mode of the zed
  ERRCODE err = zed->init(
      MODE::QUALITY, -1, true); // need quite a powerful graphic card in QUALITY

  // ERRCODE display
  cout << errcode2str(err) << endl;

  // Quit if an error occurred
  if (err != SUCCESS) {
    delete zed;
    return 1;
  }

  // print on screen the keys that can be used
  bool printHelp = false;
  std::string helpString =
      "[p] increase distance, [m] decrease distance, [q] quit";

  // get the focale and the baseline of the zed
  float fx =
      zed->getParameters()->LeftCam.fx; // here we work with the right camera

  // get width and height of the ZED images
  int width = zed->getImageSize().width;
  int height = zed->getImageSize().height;

  // create and alloc GPU memory for the disparity matrix
  Mat disparityRightGPU;
  disparityRightGPU.data = (unsigned char *)nppiMalloc_32f_C1(
      width, height, &disparityRightGPU.step);
  disparityRightGPU.setUp(width, height, 1, sl::zed::FLOAT, GPU);

  // create and alloc GPU memory for the depth matrix
  Mat depthRightGPU;
  depthRightGPU.data =
      (unsigned char *)nppiMalloc_32f_C1(width, height, &depthRightGPU.step);
  depthRightGPU.setUp(width, height, 1, sl::zed::FLOAT, GPU);

  // create and alloc GPU memory for the image matrix
  Mat imageDisplayGPU;
  imageDisplayGPU.data =
      (unsigned char *)nppiMalloc_8u_C4(width, height, &imageDisplayGPU.step);
  imageDisplayGPU.setUp(width, height, 4, sl::zed::UCHAR, GPU);

  // create a CPU image for display purpose
  cv::Mat imageDisplay(height, width, CV_8UC4);

  float depthMax = 6.; // Meter
  bool depthMaxAsChanged = true;

  char key = ' ';

  // launch a loop
  bool run = true;
  while (run) {

    // Grab the current images and compute the disparity
    bool res = zed->grab(RAW);

    // get the right image
    // !! WARNING !! this is not a copy, here we work with the data allocated by
    // the zed object
    // this can be done ONLY if we call ONE time this methode before the next
    // grab, make a copy if you want to get multiple IMAGE
    Mat imageRightGPU = zed->getView_gpu(STEREO_RIGHT);

    // get the disparity
    // !! WARNING !! this is not a copy, here we work with the data allocated by
    // the zed object
    // this can be done ONLY if we call ONE time this methode before the next
    // grab, make a copy if you want to get multiple MEASURE
    Mat disparityGPU = zed->retrieveMeasure_gpu(DISPARITY);

    //  Call the cuda function that convert the disparity from left to right
    cuConvertDisparityLeft2Right(disparityGPU, disparityRightGPU);

    // Call the cuda function that convert disparity to depth
    cuConvertDisparity2Depth(disparityRightGPU, depthRightGPU, fx, baseline);

    // Call the cuda function that convert depth to color and merge it with the
    // current right image
    cuOverlayImageAndDepth(depthRightGPU, imageRightGPU, imageDisplayGPU,
                           depthMax);

    // Copy the processed image frome the GPU to the CPU for display
    cudaMemcpy2D((uchar *)imageDisplay.data, imageDisplay.step,
                 (Npp8u *)imageDisplayGPU.data, imageDisplayGPU.step,
                 imageDisplayGPU.getWidthByte(), imageDisplayGPU.height,
                 cudaMemcpyDeviceToHost);

    if (printHelp) // write help text on the image if needed
      cv::putText(imageDisplay, helpString, cv::Point(20, 20),
                  CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(111, 111, 111, 255),
                  2);

    // display the result
    cv::imshow("Image right Overlay", imageDisplay);
    key = cv::waitKey(20);

    switch (key) // handle the pressed key
    {
    case 'q': // close the program
    case 'Q':
      run = false;
      break;

    case 'p': // increase the distance threshold
    case 'P':
      depthMax += 1;
      depthMaxAsChanged = true;
      break;

    case 'm': // decrease the distance threshold
    case 'M':
      depthMax = (depthMax > 1 ? depthMax - 1 : 1);
      depthMaxAsChanged = true;
      break;

    case 'h': // print help
    case 'H':
      printHelp = !printHelp;
      cout << helpString << endl;
      break;
    default:
      break;
    }

    if (depthMaxAsChanged) {
      cout << "New distance max " << depthMax << "m" << endl;
      depthMaxAsChanged = false;
    }
  }

  // free all the allocated memory before quit
  imageDisplay.release();
  disparityRightGPU.deallocate();
  depthRightGPU.deallocate();
  imageDisplayGPU.deallocate();
  delete zed;

  return 0;
}
