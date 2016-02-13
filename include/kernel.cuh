#include "zed/Mat.hpp"
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <algorithm>
#include "npp.h"
#include "device_functions.h"
#include <stdint.h>

// CUDA Fuction :: compute the disparity of the right image
void cuConvertDisparityLeft2Right(sl::zed::Mat &disparityLeft, sl::zed::Mat &disparityRight);

// CUDA Fuction :: compute the depth from the disparity
void cuConvertDisparity2Depth(sl::zed::Mat &disparity, sl::zed::Mat &depth, float fx, float baseline);

// CUDA Function :: convert the depth into color and overlay it with the current right image
void cuOverlayImageAndDepth(sl::zed::Mat &depth, sl::zed::Mat &imageIn, sl::zed::Mat &imageOut, float depthMax);