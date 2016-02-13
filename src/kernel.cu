#include "kernel.cuh"

/*________________________________________________________* 
*														  *
*   		CUDA KERNELS AND ASSOCIATED FUNCTIONS		  *
*														  *
*_________________________________________________________*/


// Kernel :: gives the disparity for the right image based on the disparity of the left image
__global__ void _convertDisparity(float *disparityLeft, int32_t *disparityRight, unsigned int width, unsigned int height, unsigned int leftStep, unsigned int rightStep)
{
	// get the position of the current pixel
	int x_local = blockIdx.x * blockDim.x + threadIdx.x;
	int y_local = blockIdx.y * blockDim.y + threadIdx.y;

	// exit if the pixel is out of the size of the image
	if (x_local >= width || y_local >= height) return;
	
	// Get the current disparity
	float disp_L = disparityLeft[y_local * leftStep + x_local];

	// given our computation convention the position in the associated pixel in the right image is given by :
	int32_t x_shifted_value = x_local + disp_L + 0.5f; // the position is rounded

	// check if the right pixel is inside the image
	bool test_bord = x_shifted_value >= 0 && x_shifted_value < width;

	// reject outside pixels and occlusion values
	if ((disp_L < 0) && test_bord){
		// change the type of the disparity, from float to int32_t, to be able to use the CUDA atomicMin function
		int32_t d_ = 0x80000000 - *(int32_t*)(&disp_L);
		// many pixels can be projected at the same location, we need to keep the closest one
		atomicMin(&disparityRight[x_shifted_value + y_local*rightStep], d_);
	}
}

// Kernel :: convert the disparity values from uint32_t to float
__global__ void _convertInt2Float(float *disparityRight, unsigned int width, unsigned int height, unsigned int rightStep)
{
	// get the position of the current pixel
	int x_local = blockIdx.x * blockDim.x + threadIdx.x;
	int y_local = blockIdx.y * blockDim.y + threadIdx.y;

	// exit if the pixel is out of the size of the image
	if (x_local >= width || y_local >= height) return;

	// get the value
	float disp_R = disparityRight[y_local * rightStep + x_local];

	// apply inverse transformation to the value
	uint32_t tmp = (0x80000000 - *(int32_t*)(&disp_R));
	
	// cast uint32_t into float
	disparityRight[y_local * rightStep + x_local] = *(float *)&tmp;
}

// Function :: gives the disparity of the right image based on the disparity of the left image
void cuConvertDisparityLeft2Right(sl::zed::Mat &disparityLeft, sl::zed::Mat &disparityRight)
{
	// get the image size
	unsigned int width = disparityLeft.width;
	unsigned int height = disparityLeft.height;

	// define the block dimension for the parallele computation
	dim3 dimGrid, dimBlock;
	dimBlock.x = 32;
	dimBlock.y = 8;

	dimGrid.x = ceill(width / (float)dimBlock.x);
	dimGrid.y = ceill(height / (float)dimBlock.y);
	
	// set the initial right disparity to a value greater than zero (means occlusion)
	NppiSize size; size.width = width; size.height = height;
	nppiSet_32f_C1R(5000, (float *)disparityRight.data, disparityRight.step, size);

	// call the kernel
	_convertDisparity<<<dimGrid, dimBlock>>>((float *)disparityLeft.data, (int32_t *)disparityRight.data, width, height, disparityLeft.step / sizeof(float), disparityRight.step / sizeof(int32_t));

	// call the kernel
	_convertInt2Float<<<dimGrid, dimBlock>>>((float *)disparityRight.data, width, height, disparityRight.step / sizeof(float));
}

// Kernel :: compute the depth from the disparity
__global__ void _disp2Depth(float *disparity, float *depth, float fx, float baseline, unsigned int width, unsigned int height, unsigned int disparityStep, unsigned int depthStep)
{
	// get the position of the current pixel
	int x_local = blockIdx.x * blockDim.x + threadIdx.x;
	int y_local = blockIdx.y * blockDim.y + threadIdx.y;

	// exit if the pixel is out of the size of the image
	if (x_local >= width || y_local >= height) return;
	
	// get the disparity of the current pixel
	float disp = disparity[y_local * disparityStep + x_local];

	// given our convention the disparity is always negative
	if (disp < 0){
		// the convertion is given by the formula DEPTH = FOCAL x BASELINE / DISPARITY
		// as our convention gives negative values for the disparity we need to multiply it by -1
		// in this sample we use depth in meter, so we need to convert mm in m (/ 1000)
		depth[y_local * depthStep + x_local] = (fx * baseline / disp) * -0.001;
	}
	else // the disparity is positive if it can not be estimated
		depth[y_local * depthStep + x_local] = -999.f;
}

// Fuction :: compute the depth from the disparity
void cuConvertDisparity2Depth(sl::zed::Mat &disparity, sl::zed::Mat &depth, float fx, float baseline)
{
	// get the image size
	unsigned int width = disparity.width;
	unsigned int height = disparity.height;

	// define the block dimension for the parallele computation
	dim3 dimGrid, dimBlock;
	dimBlock.x = 32;
	dimBlock.y = 8;

	dimGrid.x = ceill(width / (float)dimBlock.x);
	dimGrid.y = ceill(height / (float)dimBlock.y);

	// call the kernel
	_disp2Depth<<<dimGrid, dimBlock>>>((float *)disparity.data, (float *)depth.data, fx, baseline, width, height, disparity.step / sizeof(float), depth.step / sizeof(float));
}

// device Kernel (can only be call by a kernel) :: merge to color
inline __device__ unsigned char __mergeColor(unsigned char c1, unsigned char c2){

	return  (unsigned char)__float2int_rn((c1 + c2*3) / 4.);
}

// Kernel :: convert depth to color and merge it with the current right image
__global__ void _overlayImageDepth(float* depth, uchar4 * imageIn, uchar4 * imageOut, float depthMax, float scaleColor,
	unsigned int width, unsigned int height, unsigned int depthStep, unsigned int imInStep, unsigned int imOutStep)
{
	// get the position of the current pixel
	int x_local = blockIdx.x * blockDim.x + threadIdx.x;
	int y_local = blockIdx.y * blockDim.y + threadIdx.y;

	// exit if the pixel is out of the size of the image
	if (x_local >= width || y_local >= height) return;

	// get the depth of the current pixel
	float D = depth[y_local * depthStep + x_local];
	
	unsigned char depthColor;

	// the depth is strickly positive, if not it means that the depth can not be computed
	// the depth should be below the maximum depth
	if ((D > 0) && (D < depthMax)){
		// Scale the depth value to fit the interval [0, 255] of the color
		depthColor = 255. - (D * scaleColor);
	}else // if false : replace current pixel by the pixel of the mask
		depthColor = 0;

	// get the color of the current pixel
	uchar4 colorIn = imageIn[x_local + y_local * imInStep];
	
	// merge the color of the depth and the current image
	imageOut[x_local + y_local * imOutStep] = make_uchar4(__mergeColor(colorIn.x, depthColor), __mergeColor(colorIn.y, depthColor), __mergeColor(colorIn.z, depthColor), 255);

}

// Function :: convert depth to color and merge it with the current right image
void cuOverlayImageAndDepth(sl::zed::Mat &depth, sl::zed::Mat &imageIn, sl::zed::Mat &imageOut, float depthMax)
{
	// get the image size
	unsigned int width = depth.width;
	unsigned int height = depth.height;

	// define the block dimension for the parallele computation
	dim3 dimGrid, dimBlock;
	dimBlock.x = 32;
	dimBlock.y = 8;

	dimGrid.x = ceill(width / (float)dimBlock.x);
	dimGrid.y = ceill(height / (float)dimBlock.y);
	
	// compute a scale factor for depth to color convertion
	float scaleColor = 255.f / depthMax;
	
	// call the kernel
	_overlayImageDepth<<<dimGrid, dimBlock>>>((float *)depth.data, (uchar4 *)imageIn.data, (uchar4 *)imageOut.data, depthMax, scaleColor, width, height,
		depth.step / sizeof(float), imageIn.step / sizeof(uchar4), imageOut.step / sizeof(uchar4));
}