#include "cudahead.h"
#include <time.h>

namespace opendip
{
    #define threadNum 1024
    #define elemsPerThread 1

    #define RESIZE_CALL_NUM 1000
    
    uchar* deviceDataResized; 
    uchar* deviceData;
    uchar* hostOriginalImage;
    uchar* hostResizedImage;
    
    void reAllocPinned(unsigned int lengthSrc, unsigned int lengthResize, uchar* dataSource)
    {
        cudaMallocHost((void**)&hostOriginalImage, lengthSrc); // host pinned
        cudaMallocHost((void**)&hostResizedImage, lengthResize); // host pinned
        memcpy(hostOriginalImage, dataSource, lengthSrc);
    }
    
    void freePinned()
    {
        cudaFreeHost(hostOriginalImage);
        cudaFreeHost(hostResizedImage);
    }
    
    void initGPU(int w, int h, int c, uchar dataType){
        cudaMalloc((void**)&deviceDataResized, w*h*c*dataType);
        cudaMalloc((void**)&deviceData, w*h*c*dataType);
    }
    
    void deinitGPU(){
        cudaFree(deviceDataResized);
        cudaFree(deviceData);
    }
    
    __global__ void resizeBilinear_kernel(uchar* src, uchar* out,int w, int h, int c, int w2, int h2){
        int numberPixel = w2*h2;
        float x_ratio = float(w)/w2;
        float y_ratio = float(h)/h2;
        int stepOut = w2*c;
        int stepSrc = w*c;
        unsigned int threadId = blockIdx.x * threadNum*elemsPerThread + threadIdx.x*elemsPerThread;
        int shift = 0;
        while(threadId<numberPixel && shift<elemsPerThread){
            int yy = threadId/w2;
            int xx = threadId%w2;
    
            float fx = (xx + 0.5) * x_ratio - 0.5;
            float fy = (yy + 0.5) * y_ratio - 0.5;
    
            int ix = floor(fx);
            int iy = floor(fy);
    
            fy -= iy;
            if (iy < 0) {
                fy = 0, iy = 0;
            }
            if (iy > w - 2) {
                fy = 0, iy = w - 2;
            }
            
            fx -= ix;
            if (ix < 0) {
                fx = 0, ix = 0;
            }
            if (ix > w - 2) {
                fx = 0, ix = w - 2;
            }
                    
            short cbufx[2];
            cbufx[0] = (1.f - fx) * 2048;
            cbufx[1] = 2048 - cbufx[0];
    
            short cbufy[2];
            cbufy[0] = (1.f - fy) * 2048;
            cbufy[1] = 2048 - cbufy[0];
    
            for (int k = 0; k < c; ++k)
            {
                *(out + yy*stepOut + 3*xx + k) = (*(src + iy*stepSrc + 3*ix + k) * cbufx[0] * cbufy[0] + 
                                                  *(src + (iy+1)*stepSrc + 3*ix + k) * cbufx[0] * cbufy[1] + 
                                                  *(src + iy*stepSrc + 3*(ix+1) + k) * cbufx[1] * cbufy[0] + 
                                                  *(src + (iy+1)*stepSrc + 3*(ix+1) + k) * cbufx[1] * cbufy[1]) >> 22;
            }
            shift++;
            threadId++;
        }
    }
    
    uchar* resizeBilinear_gpu(int w, int h, int c, int w2, int h2){
        cudaError_t error; //store cuda error codes
        int length = w2 * h2;
    
        error = cudaMemcpy(deviceData, hostOriginalImage, w*h*c*sizeof(uchar), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMemcpy (pixels->deviceData), returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
            exit(EXIT_FAILURE);
        }
    
        dim3 threads = dim3(threadNum, 1,1); //block size 32,32,x
        dim3 blocks = dim3(w2*h2/ threadNum*elemsPerThread, 1,1);
        //printf("Blockdim.x %d\n", blocks.x);
        //printf("thrdim.x %d\n", threads.x);
    
        resizeBilinear_kernel <<<blocks, threads >>>(deviceData, deviceDataResized, w, h, c, w2, h2);
    
    
        cudaDeviceSynchronize();
        cudaMemcpy(hostResizedImage, deviceDataResized, length*c* sizeof(uchar), cudaMemcpyDeviceToHost);
    
        return hostResizedImage;
    }

    Image cudaResize(Image &src, int resize_w, int resize_h)
    {
        // gpu version
        Image dst(resize_w, resize_h, src.c);
        uchar *p_src_data = (uchar *)src.data;
        uchar *p_dst_data = (uchar *)dst.data;

        uchar* dataGpu = NULL;
        unsigned int lengthSrc= src.w* src.h* src.c * sizeof(uchar);
        unsigned int lengthResize= resize_w* resize_h* src.c * sizeof(uchar);
        reAllocPinned(lengthSrc, lengthResize, p_src_data); //allocate pinned host memory for fast cuda memcpy 

        initGPU(resize_w, resize_h, src.c);

        double cpu_startTime = clock();
        dataGpu = resizeBilinear_gpu(src.w, src.h, src.c, resize_w, resize_h); //init device
        double cpu_endTime = clock();
        double cpu_ElapseTime = ((double)(cpu_endTime - cpu_startTime)*1000 / (double)CLOCKS_PER_SEC);
        printf("Time GPU: %f ms\n", cpu_ElapseTime);

        // for (int i = 0; i < RESIZE_CALL_NUM; i++){
        //     dataGpu = resizeBilinear_gpu(src.w, src.h, src.c, resize_w, resize_h);
        // }

        memcpy(p_dst_data, dataGpu,lengthResize);
        deinitGPU();
        freePinned();  

        return dst;
    }

}
