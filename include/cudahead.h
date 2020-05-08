#ifndef ___CUDA_COMMON_H_
#define ___CUDA_COMMON_H_
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "image.h"

typedef unsigned char uchar;

namespace opendip{

//测试是否存在device设备-GPU
int cudaDeviceTest();
int getThreadNum();
//数组加法
int cudaVecAddTest(int N);
//share memory使用
int cudaStencilTest(int N);
//RGB2Gray
Image cudaOpenDipRGB2Gray(Image &src);
Image cudaConv2d(Image &src, int kernel_size);

//gpu resize
void reAllocPinned(unsigned int lengthSrc, unsigned int lengthResize, uchar* dataSource);
void freePinned();
uchar* resizeBilinear_gpu(int w, int h, int c, int w2, int h2);
void initGPU(int w, int h, int c, uchar dtype = sizeof(uchar));
void deinitGPU();

//Img Resize
Image cudaResize(Image &src, int resize_w, int resize_h);
// void cudaResize(uchar* src_data, int origin_w, int origin_h, int resize_w, int resize_h, int channel, uchar **dst_data_ptr);

//归约求和（one block）
float RedutionSum(float *array);
//归约求和（多个blocks)
float RedutionSumBlocks(float *a, float *b);

int threads_organization(void);
int threads_organization1(void);

int unique_gid_calc(void);
int unique_gid_calculation_2d(void);
int unique_gid_calculation_2d_2d(void);
int mem_transfer_test(void);
void query_device(void);
int sum_array(void);
}

#endif
