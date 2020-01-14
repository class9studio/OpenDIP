#include <iostream>
#include <cmath>

#include "common.h"
#include "algorithm.h"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
namespace opendip {
    Image Filter2D(Image &src, Matrix3d &kernel)
    {
        if(src.data == NULL || src.w < 1 ||  src.h < 1 || src.c != 1 || kernel.size() != 9)
        {
            cout << "source image invalid" <<  kernel.size() << src.c  << endl;
            return Image();
        }
        Matrix3d src_m;
        //卷积核旋转180
        Matrix3d kernel_m;
        kernel_m << kernel(2,2) , kernel(2,1), kernel(2,0),
                    kernel(1,2) , kernel(1,1), kernel(1,0),
                    kernel(0,2) , kernel(0,1), kernel(0,0);
        cout << "kernel-m" << endl;
        cout << src.h << src.w<< endl;
        cout << kernel_m << endl;
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
                src_m(0,0) = p_dst_bound_data[(j-1)*dst_bound.c*dst_bound.w + (i-1)*dst_bound.c];
                src_m(0,1) = p_dst_bound_data[(j-1)*dst_bound.c*dst_bound.w + i*dst_bound.c];
                src_m(0,2) = p_dst_bound_data[(j-1)*dst_bound.c*dst_bound.w + (i+1)*dst_bound.c];
                src_m(1,0) = p_dst_bound_data[j*dst_bound.c*dst_bound.w + (i-1)*dst_bound.c];
                src_m(1,1) = p_dst_bound_data[j*dst_bound.c*dst_bound.w + i*dst_bound.c];
                src_m(1,2) = p_dst_bound_data[j*dst_bound.c*dst_bound.w + (i+1)*dst_bound.c];
                src_m(2,0) = p_dst_bound_data[(j+1)*dst_bound.c*dst_bound.w + (i-1)*dst_bound.c];
                src_m(2,1) = p_dst_bound_data[(j+1)*dst_bound.c*dst_bound.w + i*dst_bound.c];
                src_m(2,2) = p_dst_bound_data[(j+1)*dst_bound.c*dst_bound.w + (i+1)*dst_bound.c];
                
                value = src_m(0,0)*kernel_m(0,0) + src_m(0,1)*kernel_m(0,1) + src_m(0,2)*kernel_m(0,2) + src_m(1,0)*kernel_m(1,0) + src_m(1,1)*kernel_m(1,1)  \
                        + src_m(1,2)*kernel_m(1,2) + src_m(2,0)*kernel_m(2,0) + src_m(2,1)*kernel_m(2,1) + src_m(2,2)*kernel_m(2,2);    //矩阵标量乘法
                p_dst_data[(j-1)*dst.c*dst.w + (i-1)*dst.c] = value;
            }
        }

        return dst;
    }
} //namespace opendip