#include <iostream>
#include <cmath>

#include "common.h"
#include "algorithm.h"

namespace opendip {

Image LinearInterpolation(Image &src_image, size_t resize_w, size_t resize_h)
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


}   //namespace opendip