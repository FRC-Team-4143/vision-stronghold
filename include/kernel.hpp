void cuInRange(const cv::gpu::GpuMat& src, cv::gpu::GpuMat out,
               unsigned char hue_low, unsigned char sat_low, unsigned char val_low,
               unsigned char hue_high, unsigned char sat_high, unsigned char val_high, cv::gpu::Stream& stream = cv::gpu::Stream::Null());

