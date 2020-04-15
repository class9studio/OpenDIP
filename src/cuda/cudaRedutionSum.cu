#include "cudahead.h"
#include <time.h>
#include <assert.h>
using namespace std;

namespace opendip{
    #define LENGTH 16
    #define THREADNUM 4
    #define BLOCKNUM 2
    void GetCudaCalError(cudaError err)
    {
        if (err != cudaSuccess)
        {
            cout << "Malloc fail, Stop!";
        }
        return;
    }
    __global__ void dot_prodeuct(float*aGpu, float*bGpu, float*sumGpu, int countNum)
    {
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        int total_thread_num = THREADNUM * BLOCKNUM;
        __shared__ float sData[THREADNUM]; //每个block的share memory
        int globa_id = tid + bid*blockDim.x;
        sData[tid] = 0;
        while(globa_id < countNum)
        {
            sData[tid] += aGpu[globa_id] * bGpu[globa_id];
            globa_id += total_thread_num;
        }
        __syncthreads();
        for(int i = THREADNUM / 2; i > 0; i /= 2)
        {
            if(tid < i)
            {
                sData[tid] = sData[tid] + sData[tid + i];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            sumGpu[bid] = sData[0];
        }
    }

    __global__ void sumReduced(float*aGpu, float*sumGpu, int countNum)
    {
        const int id = threadIdx.x;
        //定义一个共享内存
        __shared__ float sData[16];
        //为其赋值
        sData[id] = aGpu[id];
        //等待每个线程赋值完成
        __syncthreads();
    
        //实现归约求和
        /*
        1、每个数和当前索引加上总数一半的值相加，如果超出最大索引数就加0
        2、等到所有线程计算完毕
        3、线程数减半，直到减到1(这个1是现实中的1，计算机中的1为0)
        */
    
        for (int i = countNum /2; i > 0; i /= 2)
        {
            if (id < i)
            {
                sData[id] += sData[id + i];
            }
            __syncthreads();
        }
        if (id == 0)
        {
            sumGpu[0] = sData[0];
        }
    }

    float RedutionSum(float *a)
    {
        assert(a != NULL);
        float asum = 0;
        //定义Device上的内存
        float *aGpu = 0;
        float *sumGpu = 0;
        //为其开辟内存
        GetCudaCalError(cudaMalloc(&aGpu, 16 * sizeof(float)));
        GetCudaCalError(cudaMalloc(&sumGpu, 1 * sizeof(float)));
        //给aGpu 赋值
        cudaMemcpy(aGpu, a, 16 * sizeof(float), cudaMemcpyHostToDevice);
        //开一个block，每个block里面有16个thread
        sumReduced << <1, 16 >> > (aGpu, sumGpu,16);
        //将结果传回host
        cudaMemcpy(&asum, sumGpu, 1 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(aGpu);
        cudaFree(sumGpu);
        cout << "cuda array sum：" << asum << endl;
        float testSum = 0;
        for (int i = 0; i < 16; ++i)
        {
            testSum += a[i];
        }
        cout << "cpu array sum：" << testSum << endl;

        return asum;
    }

    float RedutionSumBlocks(float *a, float *b)
    {
        assert(a != NULL && b != NULL);
        float result[BLOCKNUM];
        float sum = 0;
        //定义Device上的内存
        float *aGpu = 0;
        float *bGpu = 0;
        float *sumGpu = 0;
        //为其开辟内存
        GetCudaCalError(cudaMalloc(&aGpu, LENGTH * sizeof(float)));
        GetCudaCalError(cudaMalloc(&bGpu, LENGTH * sizeof(float)));
        GetCudaCalError(cudaMalloc(&sumGpu, BLOCKNUM * sizeof(float)));
        //给aGpu,bGpu赋值
        cudaMemcpy(aGpu, a, LENGTH * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bGpu, b, LENGTH * sizeof(float), cudaMemcpyHostToDevice);

        //多个blcok，并且线程数是小于数组的长度的
        dot_prodeuct << <BLOCKNUM, THREADNUM >> > (aGpu, bGpu, sumGpu, LENGTH);
        //将结果传回host
        cudaMemcpy(result, sumGpu, BLOCKNUM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(aGpu);
        cudaFree(sumGpu);

        for(int i = 0; i < BLOCKNUM; i++)
        {
            sum += result[i];
        }
        return sum;
    }
} // namespace opendip