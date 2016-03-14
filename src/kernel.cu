#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/gpu/gpumat.hpp>
#include <opencv2/core/cuda_devptrs.hpp>

__global__ void _InRange(cv::gpu::PtrStepSz<uchar4> src, cv::gpu::PtrStep<uchar> out,
       unsigned char hue_low, unsigned char sat_low, unsigned char val_low,
       unsigned char hue_high, unsigned char sat_high, unsigned char val_high)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < src.cols && y < src.rows)
  {
    uchar4 v = src(y, x);
    if (v.y <= 0+50 && v.z >= 255-50)  // ignore white
  	out(y,x) = 0;
    else 
    if (hue_low <= v.x && v.x <= hue_high &&
        sat_low <= v.y && v.y <= sat_high &&
        val_low <= v.z && v.z <= val_high)
    {
      out(y, x) = 255;
    }
    else {
      out(y, x) = 0;
    }
  }
}

void cuInRange_caller(const cv::gpu::PtrStepSz<uchar4>& src, cv::gpu::PtrStep<uchar> out,
               unsigned char hue_low, unsigned char sat_low, unsigned char val_low,
               unsigned char hue_high, unsigned char sat_high, unsigned char val_high, cudaStream_t stream)
{
  dim3 block(16, 16);
  dim3 grid((src.cols + block.x - 1)/block.x, (src.rows + block.y - 1)/ block.y);
  _InRange<<<grid, block, 0, stream>>>(src, out, hue_low, sat_low, val_low, hue_high, sat_high, val_high);

  if (stream == 0)
     cudaDeviceSynchronize();
}
