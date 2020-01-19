#include <iostream>
#include <cmath>

#include "common.h"
#include "algorithm.h"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
namespace opendip {
    /*****************************************************************************
    *   Function name: EdgeDetection
    *   Description  : 单方向边缘检测
    *   Parameters   : src                  Source image name
    *                  kernel               边缘检测滤波器
    *   由于图像是离散的信号，我们可以用临近的两个像素差值来表示像素灰度值函数的导数
    *   df(x,y)/dx = (f(x,y) - f(x-1,y)) / 2
    *   譬如:
    *       x方向滤波器 [1, 0 , -1] 或者 [1, -1]
    *       y方向滤波器 [1, 0 , -1]T
    *       45°梯度方向:
    *             XY = [ 1 ,  0,           YX = [ 0 , 1,
    *                    0 , -1,                  -1, 0, 
    *                  ]                        ]
    *   另外需要注意:  经过卷积计算得像素值可能是负，需要求取绝对值
    * 
    *   Return Value : Image Type.         边缘检测输出图像
    * 
    *   Spec         :
    *   History:
    *
    *       1.  Date         : 2020-1-17
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    Image EdgeDetection(Image &src, MatrixXd &kernel)
    {
        if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 1 || 0 == kernel.size())
        {
            cout << "source image invalid" << endl;
            return Image();
        }

        return Filter2D(src, kernel);
    }
    /*****************************************************************************
    *   Function name: EdgeDetection
    *   Description  : 整幅图像的边缘检测
    *   Parameters   : src                   Source image name
    *                  kernelX               X方向边缘检测滤波器
    *                  kernelY               Y方向边缘检测滤波器
    *   Return Value : Image Type.           边缘检测输出图像
    * 
    *   Spec         :
    *                 X、Y方向得边缘滤波结果，叠加得到整幅图像得滤波结果
    *   History:
    *
    *       1.  Date         : 2020-1-17
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    Image EdgeDetection(Image &src, MatrixXd &kernelX, MatrixXd &kernelY) 
    {
        if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 1)
        {
            cout << "source image invalid" << endl;
            return Image();
        }  

        Image dstX = Filter2D(src, kernelX);
        Image dstY = Filter2D(src, kernelY);

        Image dst(src.w, src.h, src.c);
        unsigned char *p_srcX_data = (unsigned char *)dstX.data;
        unsigned char *p_srcY_data = (unsigned char *)dstY.data;
        unsigned char *p_dst_data = (unsigned char *)dst.data;
        int  value = 0;
        for(int j = 0; j < src.h; j++)
        {
            for(int i = 0; i < src.w; i++)
            {
                for(int z = 0; z < src.c; z++)
                {
                    value = p_srcX_data[j*dstX.w*dstX.c + i*dstX.c] + p_srcY_data[j*dstY.w*dstY.c + i*dstY.c];
                    p_dst_data[j*dst.w*dst.c + i*dst.c] = (value > 255 ? 255 : value);
                }
            }
        }

        return dst;
    }

    /*****************************************************************************
    *   Function name: GetSobel
    *   Description  : Sobel算子构造
    *   Parameters   : n                     Sobel算子维度
    *   Return Value : Image Type.           Sobel算子
    * 
    *   Spec         :
    *   History:
    *
    *       1.  Date         : 2020-1-17
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    static int Factorial(int num)
    {
        if(num == 0)
            return 1;
        else
            return num*Factorial(num - 1);
    }
    void GetSobel(int n, MatrixXd &sobX, MatrixXd &sobY)
    {
        int value = 0;
        MatrixXd sob(n, n);

        //先求第一列
        VectorXd sob_col(n);
        for(int  i = 0; i < n; i++)
        {
            value = Factorial(n - 1) / (Factorial(i)*Factorial(n -1 - i));
            sob_col(i) = value;
        }

        //再求第一行
        VectorXd sob_row(n);
        for(int i = 0; i < n; i++)
        {
            value = Factorial(n - 2) * (n - 1 - 2*i) / (Factorial(i)*Factorial(n -1 - i));
            sob_row(i) = value;
        }

        sobX = sob_col*sob_row.transpose();
        sobY = sob_row*sob_col.transpose();
    }

    /*****************************************************************************
    *   Function name: Sobel
    *   Description  : Sobel算子-边缘检测
    *   Parameters   : src                   输入原始图像
    *                  ksize                 Sobel算子维度n*n
    *   Return Value : Image Type.           输出检测图像
    * 
    *   Spec         :
    *         Sobel算子是一阶的梯度算子,作用: 在边缘检测的同时，对噪声具有平滑作用;
    *         3x3 sobel算子: [1, 0, -1] * [1, 2, 1]T
    *         其中:  [1, 0, -1]  ----边缘检测算子
    *                [1, 2, 1]T  ----标准平滑算子
    *             所以: Sobel具有平滑和微分的功效 
    *   History:
    *
    *       1.  Date         : 2020-1-17
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    Image Sobel(Image &src, int ksize)
    {
        if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 1 || ksize < 0)
        {
            cout << "source image invalid." << endl;
            return Image();
        }
        //获取sobel算子
        MatrixXd sobX(ksize,ksize), sobY(ksize,ksize);
        GetSobel(ksize, sobX, sobY);

        Image dstX = Filter2D(src, sobX);
        Image dstY = Filter2D(src, sobY);

        Image dst(src.w, src.h, src.c);
        unsigned char *p_srcX_data = (unsigned char *)dstX.data;
        unsigned char *p_srcY_data = (unsigned char *)dstY.data;
        unsigned char *p_dst_data = (unsigned char *)dst.data;
        int  value = 0;
        for(int j = 0; j < src.h; j++)
        {
            for(int i = 0; i < src.w; i++)
            {
                for(int z = 0; z < src.c; z++)
                {
                    value = p_srcX_data[j*dstX.w*dstX.c + i*dstX.c] + p_srcY_data[j*dstY.w*dstY.c + i*dstY.c];
                    p_dst_data[j*dst.w*dst.c + i*dst.c] = (value > 255 ? 255 : value);
                }
            }
        }

        return dst;
    }

    /*****************************************************************************
    *   Function name: Scharr
    *   Description  : Scharr算子
    *   Parameters   : src                   输入原始图像
    *   Return Value : Image Type.           输出图像
    * 
    *   Spec         :
    *                Scharr算子是对Sobel算子差异性的增强,通过将滤波器中的权重系数放大来增大像素值间的差异
    *                X:  [                                          Y:  [
    *                       -3, 0, 3,                                       -3, -10, -3,
    *                       -10,0,10,                                        0,  0 , 0,
    *                       -3, 0, 3,                                        3,  10, 3,
    *                    ]                                              ]
    * 
    *   History:
    *
    *       1.  Date         : 2020-1-18
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    Image Scharr(Image &src)
    {
        if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 1)
        {
            cout << "source image invalid." << endl;
            return Image();
        }
        MatrixXd Scharr_mX(3,3);
        Scharr_mX <<  -3, 0 , 3,
                      -10,0 ,10,
                      -3, 0 , 3;
        MatrixXd Scharr_mY(3,3);
        Scharr_mY <<  -3, -10 , -3,
                       0,  0 ,  0,
                       3,  10 , 3;  
                          
        Image dstX = Filter2D(src, Scharr_mX);
        Image dstY = Filter2D(src, Scharr_mY);

        Image dst(src.w, src.h, src.c);
        unsigned char *p_srcX_data = (unsigned char *)dstX.data;
        unsigned char *p_srcY_data = (unsigned char *)dstY.data;
        unsigned char *p_dst_data = (unsigned char *)dst.data;
        int  value = 0;
        for(int j = 0; j < src.h; j++)
        {
            for(int i = 0; i < src.w; i++)
            {
                for(int z = 0; z < src.c; z++)
                {
                    value = p_srcX_data[j*dstX.w*dstX.c + i*dstX.c] + p_srcY_data[j*dstY.w*dstY.c + i*dstY.c];
                    p_dst_data[j*dst.w*dst.c + i*dst.c] = (value > 255 ? 255 : value);
                }
            }
        }

        return dst;
    }

    /*****************************************************************************
    *   Function name: Laplacian
    *   Description  : Laplacian算子
    *   Parameters   : src                   输入原始图像
    *   Return Value : Image Type.           输出图像
    * 
    *   Spec         :
    *        Laplacian算子是一种二阶导数算子，对噪声比较敏感，因此常需要配合高斯滤波一起使用
    *        一阶微分:  df(x,y)/dx = f(x) - f(x -1)
    *        二阶微分:  d2f(x,y)/d2x = f(x+1) + f(x-1)-2f(x)
    *                  d2f(x,y)/d2x + d2f(x,y)/d2y = f(x, y-1) + f(x, y+1) + f(x-1,y) + f(x+1,y) -4f(x,y)
    * 
    *   History:
    *
    *       1.  Date         : 2020-1-19
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/    
    Image Laplacian(Image &src)
    {
        if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 1)   
        {
            cout << "source image invalid" << endl;
            return Image();
        }

        MatrixXd lap_m(3, 3);
        lap_m << 0,  1, 0,
                 1, -4, 1, 
                 0, 1,  0;
        return Filter2D(src, lap_m);
    }

} // namespce opendip