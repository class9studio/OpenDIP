#include <iostream>
#include <cmath>

#include "common.h"
#include "algorithm.h"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
namespace opendip {
    /*****************************************************************************
    *   Function name: Filter2D_Gray
    *   Description  : 灰度图像卷积
    *   Parameters   : src                  Source image name
    *                  kernel               3x3卷积核
    *   Return Value : Image Type.
    *   Spec         :
    *   History:
    *
    *       1.  Date         : 2020-1-15
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    Image Filter2D_Gray(Image &src, Matrix3d &kernel)
    {
        if(src.data == NULL || src.w < 1 ||  src.h < 1 || src.c != 1 || kernel.size() != 9)
        {
            cout << "source image invalid" << endl;
            return Image();
        }
        Matrix3d src_m;
        //卷积核旋转180
        Matrix3d kernel_m;
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
    /*****************************************************************************
    *   Function name: Filter2D_3M
    *   Description  : 图像卷积  --支持识别通道
    *   Parameters   : src                  Source image name
    *                  kernel               3x3卷积核
    *   Return Value : Image Type
    *   Spec         :
    *   History:
    *
    *       1.  Date         : 2020-1-15
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    Image Filter2D_3M(Image &src, Matrix3d &kernel)
    {
        if(src.data == NULL || src.w < 1 ||  src.h < 1 || kernel.size() != 9)
        {
            cout << "source image invalid"<< endl;
            return Image();
        }
        Matrix3d src_m;
        //卷积核旋转180
        Matrix3d kernel_m;
        kernel_m << kernel(2,2) , kernel(2,1), kernel(2,0),
                    kernel(1,2) , kernel(1,1), kernel(1,0),
                    kernel(0,2) , kernel(0,1), kernel(0,0);

        Image dst(src.w, src.h, src.c);
        Image dst_bound(src.w+2*src.c, src.h+2*src.c, src.c);
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
                for(int z = 0; z < src.c; z++)
                {
                    p_dst_bound_data[(j+1)*dst_bound.c*dst_bound.w + (i+1)*dst_bound.c + z] = p_src_data[j*src.c*src.w + i*src.c + z];
                }
            }
        }

        for(int j = 1; j < dst_bound.h - 1; j++)
        {
            for(int i = 1; i < dst_bound.w - 1; i++)
            {
                for(int z = 0; z < src.c; z++)
                {
                    src_m(0,0) = p_dst_bound_data[(j-1)*dst_bound.c*dst_bound.w + (i-1)*dst_bound.c + z];
                    src_m(0,1) = p_dst_bound_data[(j-1)*dst_bound.c*dst_bound.w + i*dst_bound.c + z];
                    src_m(0,2) = p_dst_bound_data[(j-1)*dst_bound.c*dst_bound.w + (i+1)*dst_bound.c + z];
                    src_m(1,0) = p_dst_bound_data[j*dst_bound.c*dst_bound.w + (i-1)*dst_bound.c + z];
                    src_m(1,1) = p_dst_bound_data[j*dst_bound.c*dst_bound.w + i*dst_bound.c + z];
                    src_m(1,2) = p_dst_bound_data[j*dst_bound.c*dst_bound.w + (i+1)*dst_bound.c + z];
                    src_m(2,0) = p_dst_bound_data[(j+1)*dst_bound.c*dst_bound.w + (i-1)*dst_bound.c + z];
                    src_m(2,1) = p_dst_bound_data[(j+1)*dst_bound.c*dst_bound.w + i*dst_bound.c + z];
                    src_m(2,2) = p_dst_bound_data[(j+1)*dst_bound.c*dst_bound.w + (i+1)*dst_bound.c + z];
                    
                    value = src_m(0,0)*kernel_m(0,0) + src_m(0,1)*kernel_m(0,1) + src_m(0,2)*kernel_m(0,2) + src_m(1,0)*kernel_m(1,0) + src_m(1,1)*kernel_m(1,1)  \
                            + src_m(1,2)*kernel_m(1,2) + src_m(2,0)*kernel_m(2,0) + src_m(2,1)*kernel_m(2,1) + src_m(2,2)*kernel_m(2,2);    //矩阵标量乘法
                    p_dst_data[(j-1)*dst.c*dst.w + (i-1)*dst.c + z] = value;                    
                }
            }
        }

        return dst;
    }

    /*****************************************************************************
    *   Function name: MatRotate180
    *   Description  : n*n矩阵逆时针旋转180
    *   Parameters   : m                    待旋转矩阵
    *   Return Value : MatrixXd             旋转后矩阵
    *   Spec         :
    *   History:
    *
    *       1.  Date         : 2020-1-15
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/    
    MatrixXd MatRotate180(MatrixXd m) 
    {
        int len = m.rows();
        MatrixXd res_m(len, len);

        int k = 0;
        for(int i = 0; i < len; i++)
        {   
            for(int j = 0; j < len; j++)
            {
                k = len - 1 - i;
                res_m(k, len-1-j) = m(i, j);
            }
        }
        return res_m;    
    }

    /*****************************************************************************
    *   Function name: Filter2D
    *   Description  : 图像卷积  --支持识别通道,不同核尺寸(3,5,7...)
    *   Parameters   : src                  Source image name
    *                  kernel               不同尺寸卷积核(3,5,7...)
    *   Return Value : Image Type           经过filter后的图像
    *   Spec         :
    *   History:
    *
    *       1.  Date         : 2020-1-15
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    Image Filter2D(Image &src, MatrixXd &kernel)
    {
        if(src.data == NULL || src.w < 1 ||  src.h < 1 || kernel.size() < 1)
        {
            cout << "source image invalid"<< endl;
            return Image();
        } 

        int len = kernel.rows();
        MatrixXd src_m(len, len);
        MatrixXd src_n(len, len);
        //卷积核旋转180
        MatrixXd kernel_m = MatRotate180(kernel);

        Image dst(src.w, src.h, src.c);
        Image dst_bound(src.w + (len-1)*src.c, src.h+(len-1)*src.c, src.c);
        unsigned char *p_src_data = (unsigned char*)src.data;
        unsigned char *p_dst_data = (unsigned char*)dst.data;
        unsigned char *p_dst_bound_data = (unsigned char*)dst_bound.data;
        int value = 0;
        int offset = len / 2;

        //拓宽边缘
        memset(p_dst_bound_data, 0, src.w*src.h*src.c);
        for(int j = 0; j < src.h; j++)
        {
            for(int i = 0; i < src.w; i++)
            {
                for(int z = 0; z < src.c; z++)
                {
                    p_dst_bound_data[(j+offset)*dst_bound.c*dst_bound.w + (i+offset)*dst_bound.c + z] = p_src_data[j*src.c*src.w + i*src.c + z];
                }
            }
        }

        //扫描矩阵
        for(int j = offset; j < dst_bound.h - offset; j++)
        {
            for(int i = offset; i < dst_bound.w - offset; i++)
            {
                for(int z = 0; z < src.c; z++)
                {
                    //对每一个像素点进行卷积操作
                    for(int m = 0; m < len; m++)
                    {
                        for(int n = 0;n < len; n++)
                        {
                            src_m(m, n) = p_dst_bound_data[(j-offset+m)*dst_bound.c*dst_bound.w + (i-offset+n)*dst_bound.c + z];
                        }
                    }

                    src_n = src_m.array() * kernel_m.array();  //矩阵标量乘法
                    value = src_n.sum();   
                    p_dst_data[(j-offset)*dst.c*dst.w + (i-offset)*dst.c + z] = value;                    
                }
            }
        }

        return dst;
    }

     /*****************************************************************************
    *   Function name: Blur
    *   Description  : 均值滤波  
    *   Parameters   : src                  Source image name
    *                  ksize                卷积核尺寸(3,5,7...)
    *   Return Value : Image Type           经过均值滤波后的图像
    *   Spec         :
    *       均值滤波的优点是在像素值变换趋势一致的情况下，可以将受噪声影响而突然变化的像素值修正到接近周围像素值变化的一致性下;
    *   丢失细节信息,变得更加模糊，滤波器范围越大，变模糊的效果越明显
    *   History:
    *
    *       1.  Date         : 2020-1-15
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/   
    Image Blur(Image &src, int ksize)
    {
        if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 1  || ksize%2 != 1)
        {
            cout << "source image invalid"<< endl;
            return Image();
        }         
        MatrixXd kernel = MatrixXd::Ones(ksize, ksize);
        int sum  = kernel.sum();
        if(sum == 0)
        {   
            cout << "blur kernel generate fail." << endl;
            return Image();
        }
        else
            kernel = kernel/sum;

        return Filter2D(src, kernel);
    }

} //namespace opendip