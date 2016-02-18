#include "kernel.cuh"

__global__ void _InRange(cv::gpu::PtrStepSz<uchar4> src, cv::gpu::PtrStep<uchar4> out,
                         unsigned char hue_low, unsigned char sat_low, unsigned char val_low,
                         unsigned char hue_high, unsigned char sat_high, unsigned char val_high)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < src.cols && y < src.rows)
  {
    uchar4 v = src(y, x);
    if (hue_low <= v.x && v.x <= hue_high &&
        sat_low <= v.y && v.y <= sat_high &&
        val_low <= v.z && v.z <= val_high)
    {
      out(y, x) = make_uchar4(255, 255, 255, 255);
    }
    else {
      out(y, x) = make_uchar4(0, 0, 0, 255);
    }
  }
}

void cuInRange(const cv::gpu::PtrStepSz<uchar4>& src, cv::gpu::PtrStep<uchar4> out,
               unsigned char hue_low, unsigned char sat_low, unsigned char val_low,
               unsigned char hue_high, unsigned char sat_high, unsigned char val_high)
{
  dim3 block(32, 8);
  dim3 grid((src.cols + block.x - 1)/block.x, (src.rows + block.y - 1)/ block.y);
  _InRange<<<grid, block>>>(src, out, hue_low, sat_low, val_low, hue_high, sat_high, val_high);
}
