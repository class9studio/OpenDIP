#include "cudahead.h"
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace opendip
{
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
  
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }

    //kernel interleaved pairs
    __global__ void reduction_interleaved_pairs(int * int_array, 
        int * temp_array, int size)
    {
        int tid = threadIdx.x;
        int gid = blockDim.x * blockIdx.x + threadIdx.x;

        if (gid > size)
            return;

        for (int offset = blockDim.x/ 2; offset > 0; offset = offset/2)
        {
            if (tid < offset)
            {
                int_array[gid] += int_array[gid + offset];
            }

            __syncthreads();
        }

        if (tid == 0)
        {
            temp_array[blockIdx.x] = int_array[gid];
        }
    }

    int reduction_intleaved_pairs(void)
    {
        printf("Running parallel reduction with interleaved pairs kernel \n");

        int size = 1 << 27;
        int byte_size = size * sizeof(int);
        int block_size = 128;

        int * h_input, *h_ref;
        h_input = (int*)malloc(byte_size);
        initialize(h_input, size, INIT_RANDOM);

        int cpu_result = reduction_cpu(h_input, size);

        dim3 block(block_size);
        dim3 grid(size / block.x);

        printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);

        int temp_array_byte_size = sizeof(int)* grid.x;

        h_ref = (int*)malloc(temp_array_byte_size);

        int * d_input, *d_temp;
        gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
        gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));

        gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
        gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
            cudaMemcpyHostToDevice));

        reduction_interleaved_pairs <<< grid, block >>> (d_input, d_temp, size);

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));

        int gpu_result = 0;
        for (int i = 0; i < grid.x; i++)
        {
            gpu_result += h_ref[i];
        }

        compare_results(gpu_result, cpu_result);

        gpuErrchk(cudaFree(d_input));
        gpuErrchk(cudaFree(d_temp));
        free(h_input);
        free(h_ref);

        cudaDeviceReset();
        return 0;
    }
}