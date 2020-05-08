#include "cudahead.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


namespace opendip
{

__global__ void print_threadIds()
{
	printf("threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d \n",
		threadIdx.x,threadIdx.y,threadIdx.z);
}


__global__ void print_details()
{
	printf("blockIdx.x : %d, blockIdx.y : %d, blockIdx.z : %d, blockDim.x : %d, blockDim.y : %d, gridDim.x : %d, gridDim.y :%d \n",
		blockIdx.x, blockIdx.y, blockIdx.z,blockDim.x, blockDim.y, gridDim.x, gridDim.y);
}

int threads_organization(void)
{
	int nx, ny;
	nx = 16;
	ny = 16;

	dim3 block(8,8,8);
	dim3 grid(nx/ block.x, ny/block.y);

	print_threadIds << <grid,block >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}

int threads_organization1(void)
{
    int nx, ny;
    nx = 16;
    ny = 16;

    dim3 block(8, 8);
    dim3 grid(nx / block.x, ny / block.y);

    print_details << <grid, block >> > ();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}

}
