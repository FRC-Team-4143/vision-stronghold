#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

void cuInRange_caller(const cv::cuda::PtrStepSz<uchar4>& src, cv::cuda::PtrStep<uchar> out,
               unsigned char hue_low, unsigned char sat_low, unsigned char val_low,
               unsigned char hue_high, unsigned char sat_high, unsigned char val_high);

void cuInRange(const cv::cuda::GpuMat& src, cv::cuda::GpuMat out,
               unsigned char hue_low, unsigned char sat_low, unsigned char val_low,
               unsigned char hue_high, unsigned char sat_high, unsigned char val_high)
{
	CV_Assert(src.type() == CV_8UC4);
	out.create(src.size(), CV_8UC1);
	cuInRange_caller(src, out, hue_low, sat_low, val_low, hue_high, sat_high, val_high);
}

