#include <iostream>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

#include "common.h"
#include "algorithm.h"

using namespace std;
using namespace Eigen;
namespace opendip
{
/*****************************************************************************
*   Function name: DetectHarrisCorners
*   Description  : Harris角点检测
*   Parameters   : src                  检测图像图像类
*                  alpha                响应函数的超参数(nice choice: 0.04<=alpha<=0.06)
*                  with_nms             是否需要最大值抑制
*                  threshold            响应函数值阈值比例参数(nice choice: 0.01)
*   Return Value : Image                原始图像大小，角点像素255
*   Spec         :
*         Harris角点检测有旋转不变性， 但是不具备尺寸不变性
*   History:
*
*       1.  Date         : 2020-3-7  1:50
*           Author       : YangLin
*           Modification : Created function
*****************************************************************************/
Image DetectHarrisCorners(Image &src, double alpha, bool with_nms, double threshold)
{
    assert(src.c == 1); //color img should switch to gray pic before use
    int ksize = 3;
    //目标图像，角点处像素255
    Image dst(src.w, src.h, src.c);

    //获取sobel算子
    MatrixXd sobX(ksize,ksize), sobY(ksize,ksize);
    GetSobel(ksize, sobX, sobY);

    // 计算图像I(x,y)在X和Y两个方向的梯度Ix、Iy
    Image dstX = Filter2D(src, sobX);
    Image dstY = Filter2D(src, sobY);    

    unsigned char *p_srcX_data = (unsigned char *)dstX.data;
    unsigned char *p_srcY_data = (unsigned char *)dstY.data;
    unsigned char *p_dst_data = (unsigned char *)dst.data;
    memset(p_dst_data, 0, src.w*src.h);
    // 计算图像两个方向梯度的乘积
    MatrixXd Ix2(src.h, src.w),Iy2(src.h, src.w),Ixy(src.h, src.w);
    for(int i = 0; i < src.h; i++)
    {
        for(int j = 0; j < src.w; j++)
        {
            double valx =  p_srcX_data[i*src.w + j];
            double valy =  p_srcY_data[i*src.w + j];
            Ix2(i,j) =  pow(valx,2);
            Iy2(i,j) =  pow(valy,2);
            Ixy(i,j) =  valx*valy;
        }
    }

    //使用高斯函数对I2x、I2y和Ixy进行高斯加权（取σ取1)
    MatrixXd guassKernel = GetGaussianKernel(7, 1);
    MatrixXd guassIx2 = FilterMatrix2d(Ix2, guassKernel);
    MatrixXd guassIy2 = FilterMatrix2d(Iy2, guassKernel);
    MatrixXd guassIxy = FilterMatrix2d(Ixy, guassKernel);

    //计算局部特征结果矩阵M的特征值和响应函数R(i,j)=det(M)-k(trace(M))^2
    MatrixXd cornerStrength = MatrixXd::Zero(src.h, src.w);
    for(int i = 0; i < src.h; i++)
    {
        for(int j = 0; j < src.w; j++)
        {
			double det_m = guassIx2(i,j) * guassIy2(i,j) - guassIxy(i,j) * guassIxy(i,j);
			double trace_m = guassIx2(i,j) + guassIy2(i,j);
			cornerStrength(i,j) = det_m - alpha * trace_m *trace_m;            
        }
    }

    // 将计算出响应函数的值R进行非极大值抑制，滤除一些不是角点的点，同时要满足大于设定的阈值
    double maxValue = cornerStrength.maxCoeff(); //最大值
    cout << "Max value: " << maxValue << endl;
    for(int i = 0; i < src.h; i++)
    {
        for(int j = 0; j < src.w; j++)
        {
            if(with_nms) //3x3邻域最大值抑制+阈值判断
            {
                if(cornerStrength(i,j)>maxValue*threshold)
                {
                    #if 0
                    int block_h = std::min(i+2, src.h-1) - std::max(0, i-1);
                    int block_w = std::min(j+2, src.w-1) - std::max(0, j-1);
                    if(maxValue == cornerStrength.block(std::max(0,i-1),std::max(0,i-1),block_h,block_w).maxCoeff())
                        p_dst_data[i*dst.w + j] = 255;
                    #endif
                    double temp_value = 0.0;
                    for(int m = i - 1; m < i + 2; m++)
                    {
                        for(int n = j - 1; n < j + 2; n++)
                        {
                            //抛弃超出范围的position
                            if(m>=0 && n>=0 && m < src.h && n < src.w)
                            {
                                temp_value = (cornerStrength(m,n)>temp_value) ? cornerStrength(m,n):temp_value;
                            }                                
                        }
                    }
                    if(temp_value == maxValue)
                    {
                        p_dst_data[i*dst.w + j] = 255; 
                    }
                    temp_value = 0; //update temp max value
                }
            }
            else
            {
                if(cornerStrength(i,j)>maxValue*threshold)
                    p_dst_data[i*dst.w + j] = 255;
            }
        }
    }
    
    return dst;
}

} //namespace opendip