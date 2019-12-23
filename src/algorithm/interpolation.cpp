#include <iostream>
#include <cmath>

#include "common.h"
#include "algorithm.h"

namespace opendip {


/*****************************************************************************
*   Function name: LinearInterpolation
*   Description  : 最邻近插值法(Nearest Interpolation)
*                  根据目的和源图尺寸，计算宽高比率，然后计算目的像素点对应于源像素点的位置
*   Parameters   : src_image            Source image name
*                  resize_w             width to resize
*                  resize_h             height to resize
*   Return Value : Image Type.
*   Spec         :
*   History:
*
*       1.  Date         : 2019-12-23
*           Author       : YangLin
*           Modification : Created function
*****************************************************************************/
Image LinearInterpolation(Image &src_image, int resize_w, int resize_h)
{
    if(resize_w == 0 || resize_h == 0)
    {
        ShowDebugInfo();
        Image res_image;
        return res_image;
    }
    
    Image dst_image(resize_w, resize_h, src_image.c);
    size_t src_i = 0, src_j = 0;
    float ratio_w = (float) resize_w / (float) src_image.w;
    float ratio_h = (float) resize_h / (float) src_image.h;
    unsigned char* p_src_data =(unsigned char*) src_image.data;
    unsigned char* p_dst_data =(unsigned char*) dst_image.data;

    for(size_t j = 0; j < dst_image.h; j++)
    {
        for(size_t i = 0; i < dst_image.w; i++)
        {
            src_i = std::round(i / ratio_h);
            src_j = std::round(j / ratio_w); 
            
            for(size_t z = 0; z < src_image.c; z++)
            {
                p_dst_data[j * dst_image.c * dst_image.w + dst_image.c*i + z] = p_src_data[src_j * src_image.c * src_image.w + src_image.c*src_i + z];
            }
        }
    }

    return dst_image;
}

/*****************************************************************************
*   Function name: BilinearInterpolation
*   Description  : 双线性插值法(Bilinear Interpolation)
*                  目的像素的位置同最邻近插值，例如  f(i+u,j+v) : u、v是浮点坐标 
*                 f(i+u,j+v) = f(i,j)*(1-u)*(1-v) + f(i,j+1)*(1-u)*v + f(i+1,j)*u*(1-v) + f(i+1,j+1)*u*v
*   Parameters   : src_image            Source image name
*                  resize_w             width to resize
*                  resize_h             height to resize
*   Return Value : Image Type.
*   Spec         :
*   History:
*
*       1.  Date         : 2019-12-23
*           Author       : YangLin
*           Modification : Created function
*****************************************************************************/
Image BilinearInterpolation(Image &src_image, int resize_w, int resize_h)
{
    if(resize_w == 0 || resize_h == 0)
    {
        ShowDebugInfo();
        Image res_image;
        return res_image;
    } 
    Image dst_image(resize_w, resize_h, src_image.c);
    size_t src_i = 0, src_j = 0;
    double mapped_i = 0.0, mapped_j = 0.0;
    double delta_i = 0.0, delta_j = 0.0;
    float ratio_w = (float) resize_w / (float) src_image.w;
    float ratio_h = (float) resize_h / (float) src_image.h;
    unsigned char* p_src_data =(unsigned char*) src_image.data;
    unsigned char* p_dst_data =(unsigned char*) dst_image.data;

     for(size_t j = 0; j < dst_image.h; j++)
    {
        for(size_t i = 0; i < dst_image.w; i++)
        {
            mapped_i = i / ratio_h;
            mapped_j = j / ratio_w;

            src_i = std::floor(mapped_i);
            src_j = std::floor(mapped_j); 
           
            delta_i = mapped_i - src_i;
            delta_j = mapped_j - src_j;

            if(src_i >= src_image.w - 1)
                src_i = src_i - 1;
            if(src_j >= src_image.h - 1)
                src_j >= src_j - 1;
            
            for(size_t z = 0; z < src_image.c; z++)
            {
                p_dst_data[j * dst_image.c * dst_image.w + dst_image.c*i + z] = p_src_data[src_j * src_image.c * src_image.w + src_image.c*src_i + z]*(1-delta_i)*(1-delta_j) +
                                                                                p_src_data[(src_j+1) * src_image.c * src_image.w + src_image.c*(src_i + 1) + z]*delta_i*delta_j +
                                                                                p_src_data[src_j * src_image.c * src_image.w + src_image.c*(src_i + 1) + z]*delta_i*(1-delta_j) +
                                                                                p_src_data[(src_j+1) * src_image.c * src_image.w + src_image.c*src_i + z]*(1-delta_i)*delta_j;
            }
        }
    }

    return dst_image;
}

}   //namespace opendip