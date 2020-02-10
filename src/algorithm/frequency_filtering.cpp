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
*   Function name: DFT_1D
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
void DFT_1D(double* src, Complex* dst, int size)
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
*   Function name: IDFT_1D
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
void IDFT_1D(Complex* src, Complex* dst, int size)
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

} // namespace opendip