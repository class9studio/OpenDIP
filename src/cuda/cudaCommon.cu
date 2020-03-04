#include<stdio.h>

#include "cudaCommon.h"

__global__ void kernel() 
{
    printf("hello world");
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
int cudaDeviceTest()
{
    kernel<<<1, 1>>>();

	return 0;
}

