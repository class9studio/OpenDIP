#include<stdio.h>

#include "cudahead.h"

namespace opendip{

void init_random_i(int *var, int n)
{
    int i;
    for (i = 0; i < n; i++)
        var[i] = 1;
}

/*****************************************************************************
*   Function name: cudaDeviceTest
*   Description  : 测试是否存在device设备-GPU
*   Parameters   : void
*   Return Value : None
*   Spec         :
*   History:
*
*       1.  Date         : 2020-3-1
*           Author       : YangLin
*           Modification : Created function
*****************************************************************************/
__global__ void kernel() 
{
    printf("hello world");
}
int cudaDeviceTest()
{
    kernel<<<1, 1>>>();

	return 0;
}

/*****************************************************************************
*   Function name: cudaVecAddTest
*   Description  : 加速向量数组加运算
*   Parameters   : N      数组长度
*   Return Value : int    success：0  fail: -1
*   Spec         :
*        通过cuda的加法运算，熟悉cuda程序编程的步骤:
*          1. Identity parallelism， 包括: 划分并行运算操作，分配GPU资源-线程
*          2. Write GPU Kernel
*          3. Setup the Problem: 分配内存，初始化操作等
*          4. Launch the Kernel
*          5. Copy results back from GPU
*   History:
*
*       1.  Date         : 2020-3-4
*           Author       : YangLin
*           Modification : Created function
*****************************************************************************/
__global__ void vevAdd(int N, float *a, float *b, float *c)
{
    // work idex, 在launch kernel的时候指定维度
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
      c[idx] = a[idx] + b[idx];
    }
}

int cudaVecAddTest(int N)
{
  float *a, *b, *c;
  float *devA, *devB, *devC;
  a = (float *)malloc(N*sizeof(float));
  b = (float *)malloc(N*sizeof(float));
  c = (float *)malloc(N*sizeof(float));

  //Allocate memory in GPU Globel Memory
  cudaMalloc(&devA, N*sizeof(float));
  cudaMalloc(&devB, N*sizeof(float));
  cudaMalloc(&devC, N*sizeof(float));

  memset(c, 0, N*sizeof(float));
  for (int i = 0; i < N; i++)
  {
    a[i] = 1.0;
    b[i] = 2.0;
  }

  cudaMemcpy(devA, a, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(devB, b, N*sizeof(float), cudaMemcpyHostToDevice);

  //Lanunch the GPU Kernel
  vevAdd<<<(N+255)/256, 256 >>>(N, devA, devB, devC); //number of thread blocks, shape of thread blocks

  //copy data back
  cudaMemcpy(c, devC, N * sizeof(float), cudaMemcpyDeviceToHost);

  for(int i=0;i<N;i++)  
	printf("%f\n",c[i]); 
  
  free(a);
  free(b);
  free(c);
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);

  return 0;
}

//One element per thread, using Global Memeory
//input elements are read several times, not an optimized way
#define THREADS_PER_BLOCK 10
#define BLOCK_SIZE THREADS_PER_BLOCK
#define RADIUS 3
__global__ void stencil(int *in, int *out)
{
  int globIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int value = 0;
  for(int offset = -RADIUS; offset <= RADIUS; offset++)
    value += in[globIdx + offset];
  out[globIdx] = value;
}

__global__ void stencil_share_memory(int *in, int *out)
{
  __shared__ int shared[BLOCK_SIZE + 2 * RADIUS];
  int globIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int locIdx = threadIdx.x + RADIUS;
  shared[locIdx] = in[globIdx];
  if(threadIdx.x < RADIUS)
  {
    shared[locIdx - RADIUS] = in[globIdx - RADIUS];
    shared[locIdx + BLOCK_SIZE] = in[globIdx + BLOCK_SIZE];
  }
  __syncthreads();
  int value = 0;
  for(int offset = -RADIUS; offset <= RADIUS; offset++)
    value += shared[locIdx + offset];
  out[globIdx] = value;
}

int cudaStencilTest(int N)
{
  int *in, *out;
  int *dev_in, *dev_out;
  in = (int *)malloc(N*sizeof(int));
  out = (int *)malloc(N*sizeof(int));

  init_random_i(in, N);

  //Allocate memory in GPU Globel Memory
  cudaMalloc(&dev_in,  N*sizeof(int));
  cudaMalloc(&dev_out, N*sizeof(int));

  // Copie des valeurs des variables de Host vers Device
  cudaMemcpy(dev_in, in, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_out, out, N*sizeof(int), cudaMemcpyHostToDevice);

  //Lanunch the GPU Kernel
  stencil_share_memory <<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dev_in, dev_out);
  //copy data back
  cudaMemcpy(out, dev_out, N*sizeof(int), cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; i++)
      printf("%i ---i=%d \n", out[i], i);
  

  free(in);   
  free(out);
  cudaFree(dev_in);
  cudaFree(dev_out);
  
  return 0;
}

}
