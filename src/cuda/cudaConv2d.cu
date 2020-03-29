#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include "cudahead.h"

namespace opendip{
static void HandleError(cudaError_t err, const char *file, int line)
{
    if(err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void conv2d(unsigned char*img, unsigned char *kernel, unsigned char *result,
    int width, int height, int kernel_size)
{
    int ti = threadIdx.x;
    int bi = blockIdx.x;
    int id = (bi*blockDim.x + ti);
    if(id >= width*height)
        return;
    //计算每个窗口的操作
    int row = id / width;
    int col = id % width;
    for(int i = 0; i < kernel_size; ++i)
    {
        for(int j = 0; j < kernel_size; j++)
        {
            unsigned char imgValue = 0;
            int curRow = row - kernel_size/2 + i;
            int curCol = col - kernel_size/2 + j;
            if(curCol < 0 || curRow < 0 || curRow >= height || curCol >= width)
            {
            }
            else
            {
                imgValue = img[curRow*width + curCol];
            }
            result[id] += kernel[i*kernel_size+j]*imgValue;
        }
    }
}

int getThreadNum()
{
    cudaDeviceProp prop;
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("gpu num %d\n", count);
    cudaGetDeviceProperties(&prop, 0);
    return prop.maxThreadsPerBlock;
}

Image cudaConv2d(Image &src, int kernel_size)
{
    assert(src.c == 1); // gray pic only
    Image dst(src.w, src.h, src.c);

    unsigned char *h_kernel = new unsigned char[kernel_size*kernel_size];
    //设置kernel
    for(int i = 0; i < kernel_size*kernel_size; ++i)
    {
        h_kernel[i] = i % kernel_size - 1;
    }
    unsigned char *h_img_src = (unsigned char*)src.data;
    unsigned char *h_img_dst = (unsigned char*)dst.data;

    unsigned char *d_img_src;
    unsigned char *d_img_dst;
    unsigned char *d_kernel;
	HANDLE_ERROR(cudaMalloc((void**)&d_img_src, src.h*src.w*sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc((void**)&d_img_dst, src.h*src.w*sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc((void**)&d_kernel,  kernel_size*kernel_size*sizeof(unsigned char)));

	HANDLE_ERROR(cudaMemcpy(d_img_src, h_img_src, src.h*src.w*sizeof(unsigned char), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_kernel, h_kernel, kernel_size*kernel_size*sizeof(unsigned char), cudaMemcpyHostToDevice));

    int threadNum = getThreadNum();
    int blockNum = (src.w*src.h - 0.5) / threadNum + 1;
    conv2d<<<blockNum, threadNum>>>(d_img_src, d_kernel, d_img_dst, src.w, src.h, kernel_size);

	HANDLE_ERROR(cudaMemcpy(h_img_dst, d_img_dst, src.h*src.w*sizeof(unsigned char), cudaMemcpyDeviceToHost));

    cudaFree(d_img_src);
    cudaFree(d_img_dst);
    cudaFree(d_kernel);

    return dst;
}
}