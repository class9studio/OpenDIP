#ifndef ___CUDA_COMMON_H_
#define ___CUDA_COMMON_H_

//测试是否存在device设备-GPU
int cudaDeviceTest();
//数组加法
int cudaVecAddTest(int N);
//share memory使用
int cudaStencilTest(int N);

#endif

