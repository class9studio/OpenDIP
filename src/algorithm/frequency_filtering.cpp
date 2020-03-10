#include <iostream>
#include <cmath>
#include <vector>
 #include<cfloat> 
 #include<limits> 

#include "common.h"
#include "algorithm.h"
/*
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;    //图库matplotlib-cpp头文件
*/
using namespace std;

namespace opendip {
//Complex的函数
Complex operator + (Complex a, Complex b) { return Complex(a.r + b.r, a.i + b.i); }
Complex operator - (Complex a, Complex b) { return Complex(a.r - b.r, a.i - b.i); }
Complex operator * (Complex a, Complex b) { return Complex(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r); }
istream& operator >> (istream &in, Complex &a) { in >> a.r >> a.i; return in; }
ostream& operator << (ostream &out, Complex &a) { out << a.r <<" + "<< a.i <<" i "; return out; }

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
void DFT2D(double *src,Complex *dst,int size_w,int size_h)
{
    for(int u = 0; u < size_w; u++)
    {
        for(int v = 0; v < size_h; v++)
        {
            double real=0.0;
            double imagin=0.0;
            for(int i = 0; i < size_w; i++)
            {
                for(int j = 0; j < size_h; j++)
                {
                    double I=src[i*size_w+j];
                    double x=OPENDIP_PI*2*((double)i*u/(double)size_w+(double)j*v/(double)size_h);
                    real+=cos(x)*I;
                    imagin+=-sin(x)*I;

                }
            }
            dst[u*size_w+v].r = real;
            dst[u*size_w+v].i = imagin;
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
void IDFT2D(Complex *src,Complex *dst,int size_w,int size_h)
{
    for(int i = 0; i < size_w; i++)
    {
        for(int j = 0; j < size_h; j++)
        {
            double real=0.0;
            double imagin=0.0;
            for(int u = 0; u < size_w; u++)
            {
                for(int v = 0; v < size_h; v++)
                {
                    double R=src[u*size_w+v].r;
                    double I=src[u*size_w+v].i;
                    double x=OPENDIP_PI*2*((double)i*u/(double)size_w+(double)j*v/(double)size_h);
                    real+=R*cos(x)-I*sin(x);
                    imagin+=I*cos(x)+R*sin(x);

                }
            }
            dst[i*size_w+j].r = (1./(size_w*size_h))*real;
            dst[i*size_w+j].i = (1./(size_w*size_h))*imagin;
        }
    }
}

static int FFTComplex_remap(Complex *src, int size_n)
{
    if(size_n==1)
        return 0;
    Complex *temp = new Complex[size_n]();

    for(int i = 0; i < size_n; i++)
        if(i%2==0)
            temp[i/2] = src[i];
        else
            temp[(size_n+i)/2] = src[i];

    for(int i = 0; i < size_n; i++)
        src[i] = temp[i];
    delete[] temp;
    FFTComplex_remap(src, size_n/2);
    FFTComplex_remap(src+size_n/2, size_n/2);
    return 1;
}
//计算旋转因子WN
static void getWN(double n, double size_n, Complex *dst)
{
    double x=2.0*OPENDIP_PI*n/size_n;
    dst->i = -sin(x);
    dst->r = cos(x);
}

static int isBase2(int size_n)
{
    int k=size_n;
    int z=0;
    while (k/=2)
        z++;
    k=z;
    if(size_n!=(1<<k))
        return -1;
    else
        return k;
}

/*****************************************************************************
*   Function name: FFT1D
*   Description  : 单变量快速傅里叶变换
*   Parameters   : src   			   单变量频域复数数组
*                  dst                 生成的频域复数数组   
*                  size                数组长度
*   Return Value : void                           
*   History:
*
*       1.  Date         : 2020-2-11
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void FFT1D(Complex *src, Complex *dst, int size)
{
    assert(src != NULL);
    int k = size;
    int z = 0;
    while (k/=2) 
        z++;
    k = z;

    if(size != (1<<k))
        exit(0);
    Complex *src_com = new Complex[size]();
    if(src_com == NULL)
        exit(0);
    for(int i = 0; i < size; i++)
    {
        src_com[i] = src[i];
    }
    FFTComplex_remap(src_com, size);
    for(int i = 0; i < k; i++)
    {
        z=0;
        for(int j = 0; j < size; j++)
        {
            if((j/(1<<i))%2==1)
            {
                Complex wn;
                getWN(z, size, &wn);
                src_com[j] = src_com[j] * wn;
                z+=1<<(k-i-1);
                Complex temp;
                int neighbour = j-(1<<(i));
                temp.r = src_com[neighbour].r;
                temp.i = src_com[neighbour].i;

                src_com[neighbour] = temp + src_com[j];
                src_com[j] = temp - src_com[j];
            }
            else
                z=0;
        }
    }

    for(int i = 0; i < size; i++)
    {
        dst[i] = src_com[i];
    }
    delete[] src_com;
}

/*****************************************************************************
*   Function name: IFFT1D
*   Description  : 单变量快速傅里叶反变换
*   Parameters   : src   			   频域复数数组
*                  dst                 恢复的频域复数数组   
*                  size                数组长度
*   Return Value : void                           
*   History:
*
*       1.  Date         : 2020-2-11
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void IFFT1D(Complex *src, Complex *dst, int size)
{
    for(int i = 0; i < size; i++)
        src[i].i = -src[i].i;
    FFTComplex_remap(src, size);
    int z,k;
    if((z=isBase2(size))!=-1)
        k=isBase2(size);
    else
        exit(0);
    for(int i = 0; i < k; i++)
    {
        z=0;
        for(int j = 0; j < size; j++)
        {
            if((j/(1<<i))%2==1){
                Complex wn;
                getWN(z, size, &wn);
                src[j] = src[j] * wn;

                z+=1<<(k-i-1);
                Complex temp;
                int neighbour=j-(1<<(i));
    
                temp = src[neighbour];
                src[neighbour] = temp + src[j];
                src[j] = temp - src[j];
            }
            else
                z=0;
        }

    }
    for(int i = 0; i < size; i++)
    {
        dst[i].i = (1./size)*src[i].i;
        dst[i].r =(1./size)*src[i].r;
    }
}

static void ColumnVector(Complex *src, Complex *dst, int size_w, int size_h)
{
    for(int i = 0; i < size_h; i++)
        dst[i] = src[size_w*i];
}

static void IColumnVector(Complex *src, Complex *dst, int size_w, int size_h)
{
    for(int i = 0; i < size_h; i++)
        dst[size_w*i] = src[i];
}

static void RealFFT(double *src, Complex *dst, int size_n)
{
    int k=size_n;
    int z=0;
    while (k/=2)
        z++;
    k=z;
    if(size_n!=(1<<k))
        exit(0);
    Complex *src_com = new Complex[size_n]();
    if(src_com==NULL)
        exit(0);
    for(int i = 0; i < size_n; i++)
    {
        src_com[i].r = src[i];
        src_com[i].i = 0;
    }
    FFTComplex_remap(src_com, size_n);
    for(int i=0; i < k; i++)
    {
        z=0;
        for(int j = 0; j < size_n; j++)
        {
            if((j/(1<<i))%2==1)
            {
                Complex wn;
                getWN(z, size_n, &wn);
                src_com[j] = src_com[j]*wn;
                
                z+=1<<(k-i-1);
                Complex temp;
                int neighbour=j-(1<<(i));
                temp.r = src_com[neighbour].r;
                temp.i = src_com[neighbour].i;

                src_com[neighbour] = temp + src_com[j];
                src_com[j] = temp + src_com[j];
            }
            else
                z=0;
        }
    }

    for(int i=0;i<size_n;i++)
        dst[i] = src_com[i];
    
    delete[] src_com;
}

/*****************************************************************************
*   Function name: FFT2D
*   Description  : 二维快速傅里叶变换
*   Parameters   : src   			   二维变量数组
*                  dst                 生成的频域复数数组   
*                  size_w              二维数组宽
*                  size_h              二维数组高
*   Return Value : void  
*   Spec:
*       二维FFT的是实现方法是先对行做FFT将结果放回该行，然后再对列做FFT结果放在该列                         
*   History:
*
*       1.  Date         : 2020-2-11
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void FFT2D(double *src, Complex *dst, int size_w, int size_h)
{
    if(isBase2(size_w)==-1 || isBase2(size_h)==-1)
        return;
    Complex *temp = new Complex[size_h*size_w]();
    if(temp==NULL)
        return;
    for(int i=0;i<size_h;i++)
    {
        RealFFT(&src[size_w*i], &temp[size_w*i], size_w);
    }

    Complex *Column = new Complex[size_h]();
    if(Column==NULL)
        return;
    for(int i=0;i<size_w;i++){
        ColumnVector(&temp[i], Column, size_w, size_h);
        FFT1D(Column, Column, size_h);
        IColumnVector(Column, &temp[i], size_w, size_h);
    }

    for(int i=0;i<size_h*size_w;i++)
        dst[i] = temp[i];
    delete[] temp;
    delete[] Column;
}

/*****************************************************************************
*   Function name: IFFT2D
*   Description  : 二维快速傅里叶反变换
*   Parameters   : src   			   二维频域复数数组
*                  dst                 恢复的频域复数数组   
*                  size_w              二维数组宽
*                  size_h              二维数组高
*   Return Value : void                           
*   History:
*
*       1.  Date         : 2020-2-11
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void IFFT2D(Complex *src,Complex *dst,int size_w,int size_h)
{
    if(isBase2(size_w)==-1||isBase2(size_h)==-1)
        return;

    Complex *temp = new Complex[size_h*size_w]();
    if(temp==NULL)
        return;
 
    Complex *Column = new Complex[size_h]();
    if(Column==NULL)
        return;

    for(int i = 0; i < size_w; i++)
    {
        ColumnVector(&src[i], Column, size_w, size_h);
        IFFT1D(Column, Column, size_h);
        IColumnVector(Column, &src[i], size_w, size_h);
    }

    for(int i = 0; i < size_w*size_h; i++)
        src[i].i = -src[i].i;
    for(int i = 0; i < size_h; i++)
    {
        IFFT1D(&src[size_w*i], &temp[size_w*i], size_w);
    }

    for(int i=0;i<size_h*size_w;i++)
        dst[i] = temp[i];

    delete[] temp;
    delete[] Column;
}

/*****************************************************************************
*   Function name: FFT_Shift
*   Description  : 图像中心化
*   Parameters   : src   			   图像数据(double数据类型存储)  
*                  size_w              二维数组宽
*                  size_h              二维数组高
*   Return Value : void   
*         图像傅里叶变换前，先进行平移Shift(图像高频-白色，聚集在图像中心，否则在图像的四个角落聚集)
*         图像傅里叶反变换后，再进行图像Shift， 图像复原                   
*   History:
*       
*       1.  Date         : 2020-2-13
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void FFT_Shift(double *src, int size_w, int size_h)
{
    for(int j = 0; j < size_h; j++)
    {
        for(int i = 0; i < size_w; i++)
        {
            if((i+j)%2)
                src[j*size_w+i]=-src[j*size_w+i];
        }
    }
}

/*****************************************************************************
*   Function name: ImgFFT
*   Description  : 图像傅里叶变换
*   Parameters   : src   			   原始图像  
*                  dst                 生成频域复数数组
*   Return Value : void                     
*   History:
*       
*       1.  Date         : 2020-2-13
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void ImgFFT(Image &src, Complex *dst)
{
    //支持灰度图像
    assert(src.c == 1);
    unsigned char *src_data = (unsigned char *)src.data;
    double *img_src = new double[src.w*src.h]();
    for(int j = 0; j < src.h; j++)
    {
        for(int i = 0; i < src.w; i++)
        {
            img_src[j*src.w*src.c+i*src.c] = src_data[j*src.w*src.c+i*src.c];
        }
    }
    FFT_Shift(img_src, src.w, src.h);
    FFT2D(img_src, dst, src.w, src.h);

    delete[] img_src;
}

/*****************************************************************************
*   Function name: ImgIFFT
*   Description  : 图像傅里叶反变换
*   Parameters   : src   			   带还原频域复数数组  
*                  size_w              频域复数数组宽长
*                  size_h              频域复数数组高长
*   Return Value : Image               还原图像                     
*   History:
*       
*       1.  Date         : 2020-2-13
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image ImgIFFT(Complex *src, int size_w, int size_h)
{
    Image dst(size_w, size_h, 1);
    unsigned char *dst_data = (unsigned char *)dst.data;

    Complex *temp_c = new Complex[size_w*size_h]();
    if(temp_c == NULL)
        exit(0);
    for(int i = 0; i < size_w*size_h; i++)
    {
        temp_c[i] = src[i];
    }
    Complex *temp = new Complex[size_w*size_h]();
    if(temp == NULL)
        exit(0);
    
    double *temp_d = new double[size_w*size_h]();
    if(temp_d == NULL)
        exit(0);
    IFFT2D(temp_c, temp, size_w, size_h);

    for(int j = 0; j < size_h; j++)
    {
        for(int i = 0; i < size_w; i++)
        {
            temp_d[j*size_w+i] = temp[j*size_w+i].r;
        }
    }

    FFT_Shift(temp_d, size_w, size_h);
    for(int j = 0; j < size_h; j++)
    {
        for(int i = 0; i < size_w; i++)
        {
            dst_data[j*size_w+i] = temp_d[j*size_w+i];
        }
    }

    delete[] temp;
    delete[] temp_c;
    delete[] temp_d;

    return dst;
}

/*****************************************************************************
*   Function name: Nomalsize
*   Description  : 幅度谱归一化
*   Parameters   : src   			   复数幅度数组  
*                  dst   			   变换到[0~255]幅度数组
*                  size_w              数组宽长
*                  size_h              数组高长
*   Return Value : void                     
*   History:
*       
*       1.  Date         : 2020-2-13
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void Nomalsize(double *src,double *dst,int size_w,int size_h)
{
    double  max=0.0,min=DBL_MAX;
    for(int i = 0; i < size_w*size_h; i++)
    {
        max = src[i]>max?src[i]:max;
        min = src[i]<min?src[i]:min;
    }
    double step = 255.0/(max-min);

    for(int i = 0; i < size_w*size_h; i++)
    {
        dst[i]=(src[i]-min)*step;
        dst[i]*=45.9*log((double)(1+dst[i]));
    }
}

/*****************************************************************************
*   Function name: getAmplitudespectrum
*   Description  : 通过复数数组得到频谱图像
*   Parameters   : src   			   复数幅度数组  
*                  size_w              数组宽长
*                  size_h              数组高长
*   Return Value : Image               生成频谱图像            
*   History:
*       
*       1.  Date         : 2020-2-13
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image getAmplitudespectrum(Complex *src, int size_w, int size_h)
{
    Image dst(size_w, size_h, 1);
    unsigned char *p_dst_data = (unsigned char *)dst.data;

    double *despe = new double[size_w*size_h]();
    if(despe==NULL)
        exit(0);
    double real = 0.0;
    double imagin = 0.0;

    for(int j = 0; j < size_h; j++)
    {
        for(int i=0;i<size_w;i++)
        {
            real = src[j*size_w+i].r;
            imagin = src[j*size_w+i].i;
            despe[j*size_w+i] = sqrt(real*real+imagin*imagin);
        }
    }
    Nomalsize(despe, despe, size_w, size_h);
    for(int j = 0; j < size_h;j++)
    {
        for(int i = 0; i < size_w; i++)
        {
            p_dst_data[j*size_w+i] = despe[j*size_w+i];
        }
    }
    delete[] despe;

    return dst;
}

/*****************************************************************************
*   Function name: IdealLPFilter
*   Description  : 理想低通滤波器
*   Parameters   : filter   		   滤波器内存地址  
*                  width               宽度
*                  height              高度
*                  cut_off_frequency   截至频率
*   Return Value : void                             
*   History:
*       
*       1.  Date         : 2020-2-14
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void IdealLPFilter(double *filter, int width, int height, double cut_off_frequency)
{
    int center_x = width/2;
    int center_y = height/2;
    double distance = 0.0;
    for(int i = 0;i < width; i++)
    {
        for(int j = 0;j < height; j++)
        {
            distance = Distance(i,j,center_x,center_y);
            if(distance <= cut_off_frequency)
                filter[j*width+i]=1.0;
            else
                filter[j*width+i]=0.0;
        }
    }
}

static void MultiFilter(double *src1, Complex *src2,Complex *dst,int size)
{
    //dst(1,1)=src1(1,1)*src2(1,1);
    for(int i = 0; i < size; i++)
    {
        dst[i].r = src2[i].r*src1[i];
        dst[i].i = src2[i].i*src1[i];
    }
}

/*****************************************************************************
*   Function name: FrequencyFiltering
*   Description  : 图像频域滤波
*   Parameters   : src   		       原始图像  
*                  filter_type         滤波类型
*                  param1              滤波器函数参数1
*                  param12             滤波器函数参数2
*   Return Value : void                             
*   History:
*       
*       1.  Date         : 2020-2-14
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image FrequencyFiltering(Image &src, Frequency_Filter_Type filter_type, double param1,int param2)
{
    assert(src.c == 1);
    Image dst(src.w, src.h, 1);
    unsigned char *p_src_data = (unsigned char *)src.data;

    //将原图像扩充至2的幂次-4倍，并用黑色填充，防止周期缠绕
    Image src_ex(4*src.w, 4*src.h, 1);
    GrayImgMap src_map = GrayImgCvtMap(src);
    GrayImgMap src_ex_map = GrayImgCvtMap(src_ex);
    GrayImgMap dst_map = GrayImgCvtMap(dst);
    src_ex_map.block(0,0,src.h,src.w) = src_map;

    //产生滤波器
    double *filter = new double[src.w*src.h]();
    if(filter == NULL)
        exit(0);
    switch (filter_type)
    {
    case FRE_FILTER_ILPF:
        IdealLPFilter(filter, 4*src.w, 4*src.h, param1);
        break;
    case FRE_FILTER_BLPF:
        break;
    case FRE_FILTER_GLPF:
        break;
    default:
        break;
    }
    //FFT
    Complex *temp_complex = new Complex[4*src.w*4*src.h]();
    if(temp_complex == NULL)
        exit(0);
    ImgFFT(src_ex, temp_complex);
    //滤波
    MultiFilter(filter, temp_complex, temp_complex, 4*src.w*4*src.h);
    //IFFT
    ImgIFFT(temp_complex, 4*src.w, 4*src.h);
    //还原图像
    dst_map = src_ex_map.block(0,0,src.h,src.w);

    delete[] filter;
    delete[] temp_complex;

    return dst;
}

} // namespace opendip