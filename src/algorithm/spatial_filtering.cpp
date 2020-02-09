#include <iostream>
#include <cmath>

#include "common.h"
#include "algorithm.h"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
namespace opendip {
    /*****************************************************************************
    *   Function name: MatRotate180
    *   Description  : 任意矩阵逆时针旋转180
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
        int row = m.rows();
        int col = m.cols();
        MatrixXd res_m(row, col);

        int k = 0;
        for(int i = 0; i < row; i++)
        {   
            for(int j = 0; j < col; j++)
            {
                k = row - 1 - i;
                res_m(k, col-1-j) = m(i, j);
            }
        }
        return res_m;    
    }

    /*****************************************************************************
    *   Function name: Filter_Gray
    *   Description  : 灰度图像卷积
    *   Parameters   : src                  Source image name
    *                  kernel               卷积核
    *                  padding              填充(默认根据核大小填充)  
    *   Return Value : Image                卷积后图像
    *   Spec         :
    *   History:
    *
    *       1.  Date         : 2020-2-8
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    Image Filter2D_Gray(Image &src, MatrixXd &kernel,int padding)
    {
        assert(src.c == 1 || (kernel.rows() == kernel.cols() && kernel.rows() % 2 == 1));
        //卷积核旋转180
        int row = kernel.rows();
        int col = kernel.cols();
        if(padding == 0)
		    padding = kernel.rows() / 2;
        MatrixXd kernel_m = MatRotate180(kernel); 
        RowMatrixXc src_m(row, col);

        Image dst(src.w+padding*2-col+1, src.h+padding*2-row+1 , src.c);
        Image dst_bound(src.w + padding*2, src.h + padding*2, src.c);

        unsigned char *p_src_data = (unsigned char*)src.data;
        unsigned char *p_dst_data = (unsigned char*)dst.data;
        unsigned char *p_dst_bound_data = (unsigned char*)dst_bound.data;
        int value = 0;
        int offset_row = row / 2;
        int offset_col = col / 2;
        int pixel_val = 0;

        memset(p_dst_data, 0, dst.w*dst.h*dst.c);
        memset(p_dst_bound_data, 0, dst_bound.w*dst_bound.h*dst_bound.c);
        //拓宽边缘
        vector<GrayImgMap> maps = GrayImgCvtMap(src);
        vector<GrayImgMap> maps_bound = GrayImgCvtMap(dst_bound);

        // map的加和操作会修改data的数据
        maps_bound[0].block(offset_row,offset_col,src.h, src.w) = maps[0];
        
        //扫描矩阵
        for(int j = 0; j < dst_bound.h - row + 1; j++)
        {
            for(int i = 0; i < dst_bound.w - col + 1; i++)
            {
                src_m = maps_bound[0].block(j,i,row,col);
                for(int m = 0; m < row; m++)
                {   
                    for(int n = 0; n < col; n++)
                    {
                        pixel_val = src_m(m,n);
                        value += pixel_val*kernel(m,n);
                    }
                }

                //处理卷积后的像素值可能超出[0~255]范围
                if(value > 255 || value < -255)
                {
                    value =  255;
                }  
                else
                {
                    value = abs(value);
                }
                p_dst_data[j*dst.c*dst.w + i*dst.c] = value; 
                value = 0;
            }
        }

        return dst;
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
        assert(src.c == 1 || src.c == 3 || (kernel.rows() == kernel.cols() && kernel.rows() % 2 == 1));

        int row = kernel.rows();
        int col = kernel.cols();

        //卷积核旋转180
        MatrixXd kernel_m = MatRotate180(kernel);

        Image dst(src.w, src.h, src.c);
        Image dst_bound(src.w + col - 1, src.h + row - 1, src.c);
        unsigned char *p_src_data = (unsigned char*)src.data;
        unsigned char *p_dst_data = (unsigned char*)dst.data;
        unsigned char *p_dst_bound_data = (unsigned char*)dst_bound.data;
        int value = 0;
        int offset_row = row / 2;
        int offset_col = col / 2;
        int pixel_val = 0;

        //拓宽边缘
        memset(p_dst_bound_data, 0, dst_bound.w*dst_bound.h*dst_bound.c);
        for(int j = 0; j < src.h; j++)
        {
            for(int i = 0; i < src.w; i++)
            {
                for(int z = 0; z < src.c; z++)
                {
                    p_dst_bound_data[(j+offset_row)*dst_bound.c*dst_bound.w + (i+offset_col)*dst_bound.c + z] = p_src_data[j*src.c*src.w + i*src.c + z];
                }
            }
        }

        //扫描矩阵
        for(int j = offset_row; j < dst_bound.h - offset_row; j++)
        {
            for(int i = offset_col; i < dst_bound.w - offset_col; i++)
            {
                for(int z = 0; z < src.c; z++)
                {
                    //对每一个像素点进行卷积操作
                    for(int m = 0; m < row; m++)
                    {
                        for(int n = 0;n < col; n++)
                        {
                            pixel_val = p_dst_bound_data[(j-offset_row+m)*dst_bound.c*dst_bound.w + (i-offset_col+n)*dst_bound.c + z];
                            value += pixel_val*kernel_m(m,n);
                        }
                    }

                    //处理卷积后的像素值可能超出[0~255]范围
                    if(value > 255 || value < -255)
                    {
                        value =  255;
                    }  
                    else
                    {
                        value = abs(value);
                    }
                    p_dst_data[(j-offset_row)*dst.c*dst.w + (i-offset_col)*dst.c + z] = value; 
                    value = 0;                   
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

    /*****************************************************************************
    *   Function name: GetGaussianKernel
    *   Description  : 二阶高斯滤波器 
    *   Parameters   : ksize                卷积核尺寸(3,5,7...)
    *                  sigma                标准差
    *   Return Value : MatrixXd             高斯卷积核
    *   Spec         :
    *       高斯滤波器考虑了像素离滤波器中心距离的影响，以滤波器中心位置为高斯分布的均值，根据高斯分布公式和每个像素离中心位置的距离计算出滤波器内每个位置的数值，
    *   从而形成高斯滤波器
    *   History:
    *
    *       1.  Date         : 2020-1-16
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/    
    MatrixXd GetGaussianKernel(int  ksize, double sigma)
    {
        MatrixXd gaussian_kernel(ksize, ksize);
        if(ksize % 2 != 1)
        {
            cout << "gaussian blur ksize invalid" << endl;
            return MatrixXd();
        }

        double sigmaX = sigma > 0 ? sigma : ((ksize-1)*0.5 - 1)*0.3 + 0.8;

        int center = ksize / 2;
        double sum = 0;

        for(int i = 0; i < ksize; i++)
        {
            for(int j = 0; j < ksize; j++)
            {
                gaussian_kernel(i,j) = (1/(2*OPENDIP_PI*sigmaX*sigmaX))*exp(-((i-center)*(i-center)+(j-center)*(j-center))/(2*sigmaX*sigmaX));
                sum += gaussian_kernel(i,j);
            }
        }

        for(int i = 0; i < ksize; i++)
        {
            for(int j = 0; j < ksize; j++)
            {
                gaussian_kernel(i,j) /= sum;
            }
        }

        return gaussian_kernel;
    }

     /*****************************************************************************
    *   Function name: GaussianBlur
    *   Description  : 高斯滤波  
    *   Parameters   : src                  Source image name
    *                  ksize                卷积核尺寸(3,5,7...)
    *                  sigma                高斯分布标准差
    *   Return Value : Image Type           经过高斯滤波后的图像
    *   Spec         :
    * 
    *   History:
    *
    *       1.  Date         : 2020-1-16
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/   
    Image GaussianBlur(Image &src, int ksize, double sigma)
    {
        if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 1  || ksize%2 != 1)
        {
            cout << "source image invalid"<< endl;
            return Image();
        }        
        MatrixXd gaussian_kernel = GetGaussianKernel(ksize, sigma);
        return Filter2D(src, gaussian_kernel);
    }

} //namespace opendip