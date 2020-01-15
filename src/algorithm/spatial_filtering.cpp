#include <iostream>
#include <cmath>

#include "common.h"
#include "algorithm.h"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
namespace opendip {
/*****************************************************************************
*   Function name: Filter2D
*   Description  : 图像的卷积,自动判断单通道还是彩色图像
*   Parameters   : src                  原始图像
*                  kernel               卷积核
*   Return Value : Image                输出图像
*   Spec         :
*   History:
*
*       1.  Date         : 2020-1-15
*           Author       : yanglin
*           Modification : Created function
*****************************************************************************/
Image Filter2D(Image &src, Matrix<unsigned char, 3, 3> &kernel)
{
    if(src.data == NULL || src.w < 1 ||  src.h < 1 || src.c != 1 || kernel.size() != 9)
        return Image();
    Matrix<unsigned char, 3, 3> src_m;
    //卷积核旋转180
    Matrix<unsigned char, 3, 3> kernel_m;
    kernel_m << kernel(2,2) , kernel(2,1), kernel(2,0),
                kernel(1,2) , kernel(1,1), kernel(1,0),
                kernel(0,2) , kernel(0,1), kernel(0,0);

    Image dst(src.w, src.h, src.c);
    Image dst_bound(src.w+2, src.h+2, src.c);
    unsigned char *p_src_data = (unsigned char*)src.data;
    unsigned char *p_dst_data = (unsigned char*)dst.data;
    unsigned char *p_dst_bound_data = (unsigned char*)dst_bound.data;
    int value = 0;
    //拓宽边缘
    memset(p_dst_bound_data, 0, src.w*src.h*src.c);
    for(int j = 0; j < src.h; j++)
    {
        for(int i = 0; i < src.w; i++)
        {
            p_dst_bound_data[(j+1)*dst_bound.c*dst_bound.w + (i+1)*dst_bound.c] = p_src_data[j*src.c*src.w + i*src.c];
        }
    }

    for(int j = 1; j < dst_bound.h - 1; j++)
    {
        for(int i = 1; i < dst_bound.w - 1; i++)
        {
            src_m(0,0) = p_dst_bound_data[(j-1)*dst.c*dst.w + (i-1)*dst.c];
            src_m(0,1) = p_dst_bound_data[(j-1)*dst.c*dst.w + i*dst.c];
            src_m(0,2) = p_dst_bound_data[(j-1)*dst.c*dst.w + (i+1)*dst.c];
            src_m(1,0) = p_dst_bound_data[j*dst.c*dst.w + (i-1)*dst.c];
            src_m(1,1) = p_dst_bound_data[j*dst.c*dst.w + i*dst.c];
            src_m(1,2) = p_dst_bound_data[j*dst.c*dst.w + (i+1)*dst.c];
            src_m(2,0) = p_dst_bound_data[(j+1)*dst.c*dst.w + (i-1)*dst.c];
            src_m(2,1) = p_dst_bound_data[(j+1)*dst.c*dst.w + i*dst.c];
            src_m(2,2) = p_dst_bound_data[(j+1)*dst.c*dst.w + (i+1)*dst.c];
            
            value = src_m(0,0)*kernel_m(0,0) + src_m(0,1)*kernel_m(0,1) + src_m(0,2)*kernel_m(0,2) + src_m(1,0)*kernel_m(1,0) + src_m(1,1)*kernel_m(1,1)  \
                    + src_m(1,2)*kernel_m(1,2) + src_m(2,0)*kernel_m(2,0) + src_m(2,1)*kernel_m(2,1) + src_m(2,2)*kernel_m(2,2);    //矩阵标量乘法
            p_dst_data[(j-1)*dst.c*dst.w + (i-1)*dst.c] = value;
        }
    }

    return dst;
}

} //namespace opendip