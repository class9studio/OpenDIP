#ifndef ___CUDA_COMMON_H_
#define ___CUDA_COMMON_H_
#include "image.h"

using namespace opendip;

//测试是否存在device设备-GPU
int cudaDeviceTest();
//数组加法
int cudaVecAddTest(int N);
//share memory使用
int cudaStencilTest(int N);
//RGB2Gray
Image cudaOpenDipRGB2Gray(Image &src);
Image cudaConv2d(Image &src, int kernel_size);
#endif
