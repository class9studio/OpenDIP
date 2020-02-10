#include <iostream>
#include <cmath>
#include <vector>

#include "common.h"
#include "algorithm.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;    //图库matplotlib-cpp头文件
using namespace std;

namespace opendip {
//Complex的函数
Complex operator + (Complex a, Complex b) { return Complex(a.r + b.r, a.i + b.i); }
Complex operator - (Complex a, Complex b) { return Complex(a.r - b.r, a.i - b.i); }
Complex operator * (Complex a, Complex b) { return Complex(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r); }

/*****************************************************************************
*   Function name: DFT1D
*   Description  : 单变量离散傅里叶变换
*   Parameters   : src   			    一维空间变量数组
*                  dst                 生成的频域复数数组   
*                  size                数组大小
*   Return Value : void                           
*   History:
*
*       1.  Date         : 2020-2-10
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void DFT1D(double* src, Complex* dst, int size)
{
    for(int m = 0; m < size; m++)
    {
        double real=0.0;
        double imagin=0.0;
        for(int n = 0; n < size; n++)
        {
            double x = OPENDIP_PI*2*m*n;
            real += src[n]*cos(x/size);
            imagin += src[n]*(-sin(x/size));

        }
        dst[m].i = imagin;
        dst[m].r = real;
    }
}

/*****************************************************************************
*   Function name: IDFT1D
*   Description  : 单变量离散傅里叶反变换
*   Parameters   : src   			    待变换频域数组
*                  dst                  生成的频域复数     
*                  size                 数组大小
*   Return Value : void                           
*   History:
*
*       1.  Date         : 2020-2-10
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void IDFT1D(Complex* src, Complex* dst, int size)
{
    for(int m = 0; m < size; m++)
    {
        double real=0.0;
        double imagin=0.0;
        for(int n = 0; n < size; n++)
        {
            double x = OPENDIP_PI*2*m*n/size;
            real += src[n].r*cos(x)-src[n].i*sin(x);
            imagin += src[n].r*sin(x)+src[n].i*cos(x);

        }
        real/=size;
        imagin/=size;
        if(dst!=NULL)
        {
            dst[m].r = real;
            dst[m].i = imagin;
        }
    }
}

/*****************************************************************************
*   Function name: DFT2D
*   Description  : 二维离散傅里叶变换
*   Parameters   : src   			   二维空间变量数组
*                  dst                 生成的频域复数数组   
*                  w                   宽长
*                  h                   高长
*   Return Value : void                           
*   History:
*
*       1.  Date         : 2020-2-10
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void DFT2D(double** src, Complex** dst, int w, int h)
{
    for(int u = 0; u < w; u++)
    {
        for(int v = 0; v < h; v++)
        {
            double real=0.0;
            double imagin=0.0;
            for(int i = 0; i < w; i++)
            {
                for(int j = 0; j < h; j++)
                {
                    double I = src[i][j];
                    double x = OPENDIP_PI*2*((double)i*u/(double)w+(double)j*v/(double)h);
                    real += cos(x)*I;
                    imagin += -sin(x)*I;

                }
            }
            dst[u][v].r = real;
            dst[u][v].i = imagin;
        }
    }
}

/*****************************************************************************
*   Function name: IDFT2D
*   Description  : 二维离散傅里叶反变换
*   Parameters   : src   			   二维频域复数数组
*                  dst                 生成的频域复数数组   
*                  w                   宽长
*                  h                   高长
*   Return Value : void                           
*   History:
*
*       1.  Date         : 2020-2-10
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void IDFT2D(Complex** src, Complex** dst, int w, int h)
{
    for(int i = 0; i < w; i++)
    {
        for(int j = 0; j < h; j++)
        {
            double real=0.0;
            double imagin=0.0;
            for(int u = 0; u < w; u++)
            {
                for(int v = 0; v < h; v++)
                {
                    double R = src[u][v].r;
                    double I = src[u][v].i;
                    double x = OPENDIP_PI*2*((double)i*u/(double)w+(double)j*v/(double)h);
                    real += R*cos(x)-I*sin(x);
                    imagin += I*cos(x)+R*sin(x);

                }
            }
            dst[i][j].r = (1./(w*h))*real;
            dst[i][j].i = (1./(w*h))*imagin;
        }
    }
}

} // namespace opendip