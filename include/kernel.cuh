#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/gpu/gpumat.hpp>

void cuInRange(const cv::gpu::PtrStepSz<uchar4>& src, cv::gpu::PtrStep<uchar> out,
               unsigned char hue_low, unsigned char sat_low, unsigned char val_low,
               unsigned char hue_high, unsigned char sat_high, unsigned char val_high);
