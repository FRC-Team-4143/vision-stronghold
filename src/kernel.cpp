#include <opencv2/gpu/stream_accessor.hpp>

void cuInRange_caller(const cv::gpu::PtrStepSz<uchar4>& src, cv::gpu::PtrStep<uchar> out,
               unsigned char hue_low, unsigned char sat_low, unsigned char val_low,
               unsigned char hue_high, unsigned char sat_high, unsigned char val_high, cudaStream_t stream);

void cuInRange(const cv::gpu::GpuMat& src, cv::gpu::GpuMat out,
               unsigned char hue_low, unsigned char sat_low, unsigned char val_low,
               unsigned char hue_high, unsigned char sat_high, unsigned char val_high, cv::gpu::Stream& stream = cv::gpu::Stream::Null())
{
	CV_Assert(src.type() == CV_8UC4);
	out.create(src.size(), CV_8UC1);
	cudaStream_t s = cv::gpu::StreamAccessor::getStream(stream);
	cuInRange_caller(src, out, hue_low, sat_low, val_low, hue_high, sat_high, val_high, s);
}

