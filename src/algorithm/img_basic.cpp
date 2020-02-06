#include <iostream>
#include <cmath>

#include "common.h"
#include "algorithm.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;    //图库matplotlib-cpp头文件

namespace opendip {
/*****************************************************************************
*   Function name: ImgShow
*   Description  : 显示图像
*   Parameters   : src_image            Source image name
*   Return Value : None
*   Spec         :
*   History:
*
*       1.  Date         : 2020-2-6
*           Author       : YangLin
*           Modification : Created function
*****************************************************************************/
void ImgShow(Image &src, string title)
{
	assert(src.c == 1 || src.c == 3);
	const unsigned char* buff = (unsigned char *)src.data;
	int h = src.h;
	int w = src.w;
	int channels = src.c;
	plt::title(title);
	if(src.c == 1)
	{
		std::map<std::string, std::string> keywords;
		keywords["cmap"] = "gray";
		plt::imshow(buff, h, w, channels, keywords);
	}
	else
	{
		plt::imshow(buff, h, w, channels);
	}
	plt::show();
}

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
        return Image();
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
            src_i = std::round(i / ratio_w);
            src_j = std::round(j / ratio_h); 
            
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
        return Image();
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
            mapped_i = i / ratio_w;
            mapped_j = j / ratio_h;

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

/*****************************************************************************
*   Function name: Split
*   Description  : sperate one channel from image
*   Parameters   : src           source image
*                  channel       channel to get(RGB->012)
*   Return Value : Image         channel image
*   Spec         : 
*   History:
*
*       1.  Date         : 2019-12-23
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image Split(Image &src, OpenDIP_Channel_Type channel)
{
	if(src.data == NULL || src.w <= 1 || src.h <= 1 || src.c < channel)
	{
		return Image();
	}

	Image img_c(src.w, src.h, 1);
	unsigned char* p_src_data =(unsigned char*) src.data;
	unsigned char* p_dst_data =(unsigned char*) img_c.data;
    for(size_t j = 0; j < src.h; j++)
    {
        for(size_t i = 0; i < src.w; i++)
        {
			p_dst_data[j * img_c.c * img_c.w + img_c.c*i] = p_src_data[j * src.c * src.w + src.c*i + channel];
        }
    }

	return img_c;
}

/*****************************************************************************
*   Function name: Split
*   Description  : sperate channels from image
*   Parameters   : src           source image
*    
*   Return Value : vector<Image>   channels image
*   Spec         : 
*   History:
*
*       1.  Date         : 2019-12-23
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
vector<Image> Split(Image &src)
{
	if(src.data == NULL || src.w <= 1 || src.h <= 1 || src.c < 1)
	{
		return vector<Image>();
	}
	vector<Image> channels;
	for(size_t c = 0; c < src.c; c++)
	{
		Image img_tmp(src.w, src.h, 1);
		channels.push_back(img_tmp);
	}

	unsigned char* p_src_data =(unsigned char*) src.data;
	unsigned char* p_dst_data = NULL;

    for(size_t j = 0; j < src.h; j++)
    {
        for(size_t i = 0; i < src.w; i++)
        {
			for(size_t z = 0;  z < src.c; z++)
			{
				p_dst_data =(unsigned char*) (channels[z].data);
				p_dst_data[j * channels[z].c * channels[z].w + channels[z].c*i] = p_src_data[j * src.c * src.w + src.c*i + z];
			}
        }
    }

	return channels;
}

/*****************************************************************************
*   Function name: Merge
*   Description  : merge channels to image
*   Parameters   : channels         channels image
*    			   num              channels numbers
*   Return Value : Image            dst image
*   Spec         : 
*   History:
*
*       1.  Date         : 2019-12-24
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image Merge(vector<Image> &channels, int num)
{
	if(num != channels.size())
	{
		return Image();
	}
	int w = channels[0].w;
	int h = channels[0].h;
	Image dst(w, h, num);

	unsigned char* p_src_data = NULL;
	unsigned char* p_dst_data =(unsigned char*) dst.data;

	for(size_t j = 0; j < h; j++)
	{
		for(size_t i = 0; i < w; i++)
		{
			for(size_t z = 0; z < num; z++)
			{
				p_src_data =(unsigned char*) channels[z].data;
				p_dst_data[j * dst.c * dst.w + dst.c*i + z] = p_src_data[j * channels[z].c * channels[z].w + channels[z].c*i];
			}
		}
	}

	return dst;
}


/*****************************************************************************
*   Function name: MeanStddev
*   Description  : mean and stddev in one channel image
*   Parameters   : src              source image
*    			   mean             均值
*                  stddev           方差
*   Return Value : None
*   Spec         : 
*   History:
*
*       1.  Date         : 2019-12-25
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void MeanStddev(Image &src, double *mean, double *stddev)
{
	if(src.c !=1 || mean == NULL || stddev == NULL)
	{
		std::cout << "invalid param" << std::endl;
		return;
	}
	double mean_res = 0.0;
	double stddev_res = 0.0;
	unsigned int *hist = new unsigned int[256]();
	double *hist_p = new double[256]();
	unsigned char *p_src_data = (unsigned char *)src.data;  
	for(size_t j = 0; j < src.h; j++)
    {
        for(size_t i = 0; i < src.w; i++)
        {
			hist[p_src_data[j * src.c * src.w + src.c*i]] += 1;
        }
    }

	for(size_t i = 0; i < 256; i++)
	{
		hist_p[i] =(double) hist[i] / (src.w*src.h);
		mean_res +=  i*hist_p[i];
	}

	for(size_t i = 0; i < 256; i++)
	{
		stddev_res += (i - mean_res)*(i - mean_res)*hist_p[i];
	}

	*mean = mean_res;
	*stddev = sqrt(stddev_res);
}

/*****************************************************************************
*   Function name: MinMaxLoc
*   Description  : max and min gray in one channel image
*   Parameters   : src              source image
*    			   min              最小灰度值
*                  max              最大灰度值
*                  min_loc          最小灰度值坐标
*                  max_loc          最大灰度值坐标
*   Return Value : None
*   Spec         : 
*   History:
*
*       1.  Date         : 2019-12-25
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void MinMaxLoc(Image &src, unsigned char *min, unsigned char *max, Point &min_loc, Point &max_loc)
{
	if(src.c != 1 || min == NULL || max == NULL)
	{
		std::cout << "invalid param" << std::endl;
		return;
	}
	unsigned char *p_src_data =(unsigned char *)src.data ;
	unsigned char min_res = p_src_data[0];
	unsigned char max_res = p_src_data[0];
	unsigned char value = 0;
	for(size_t j = 0; j < src.h; j++)
	{
		for(size_t i = 0; i < src.w; i++)
		{
			value = p_src_data[j * src.c * src.w + src.c*i];
			if(value < min_res)
			{
				min_res = value;
				min_loc.x = i;
				min_loc.y = j;
			}

			if(value > max_res)
			{
				max_res = value;
				max_loc.x = i;
				max_loc.y = j;
			}
		}
	}

	*min = min_res;
	*max = max_res;
}

/*****************************************************************************
*   Function name: ImageCvtMap
*   Function name: ImageCvtMapConst    ---read only, could't change date
*   Description  : single channel image convert to Mat format
*   Parameters   : src              image to Map
*
*   Return Value : Map              Map of image
*   Spec         : 
*   History:
*
*       1.  Date         : 2019-12-26
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
MapType ImageCvtMap(Image &src)
{
	return MapType((unsigned char *)src.data, src.h, src.w);
}

MapTypeConst ImageCvtMapConst(Image &src)
{
	return MapTypeConst((unsigned char *)src.data, src.h, src.w);
}

/*****************************************************************************
*   Function name: GetRotationMatrix2D
*   Description  : 通过旋转角度和旋转中心，返回图像旋转矩阵2x3
*   Parameters   : center            图像旋转的中心位置
*                  angle             图像旋转的角度，单位为度，正值为逆时针旋转。
*                  scale             两个轴的比例因子，可以实现旋转过程中的图像缩放，不缩放输入1
*   Return Value : Matrix            2*3的旋转矩阵
*   Spec         : 
*   History:
*
*       1.  Date         : 2019-12-31
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Matrix<double, 2, 3> GetRotationMatrix2D (Point2f center, double angle, double scale)
{
	Matrix<double, 2, 3> m;
	angle *= OPENDIP_PI/180;
	double alpha = scale * cos(angle);
	double beta = scale * sin(angle);
	m(0,0) = alpha;
	m(0,1) = beta;
	m(0,2) = (1-alpha)*center.x - beta*center.y;
	m(1,0) = -beta;
	m(1,1) = alpha;
	m(1,2) = beta*center.x + (1 - alpha)*center.y;

	return m;
}

/*****************************************************************************
*   Function name: WarpAffine
*   Description  : 仿射变换
*   Parameters   : src：			 输入图像      
*                  transform         2×3的变换矩阵
*   Return Value : Image             输出图像
*   Spec         : 
*   History:
*
*       1.  Date         : 2020-1-3
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image WarpAffine(Image &src, Matrix<double, 2, 3> transform)
{
	if(src.data == NULL || src.w <= 1 || src.h <= 1 || src.c < 1)
	{
		return Image();
	}

	Matrix3d Mat1;
    Mat1 << transform(0,0), transform(0,1), transform(0,2),
        	transform(1,0), transform(1,1), transform(1,2),
        	0, 0, 1;
	Matrix3d Mat1_inv = Mat1.inverse();
	Image img_c(src.w, src.h, src.c);
	Vector3d dst_m;
	Vector3d src_m;
	unsigned char* p_src_data =(unsigned char*) src.data;
	unsigned char* p_dst_data =(unsigned char*) img_c.data;
    for(size_t j = 0; j < src.h; j++)
    {
        for(size_t i = 0; i < src.w; i++)
        {
			dst_m(0) = i;
			dst_m(1) = j;
			dst_m(2) = 1;
			
			src_m = Mat1_inv * dst_m;
			if(src_m(0) > src.w)
			{
				src_m(0) = src.w;
			}
			if(src_m(1) > src.h)
			{
				src_m(1) = src.h;
			}

			for(size_t z = 0; z < src.c; z++)
			{
				p_dst_data[j * img_c.c * img_c.w + img_c.c*i + z] = p_src_data[(int)src_m(1) * src.c * src.w + src.c*(int)src_m(0) + z];
			}
        }
    }

	return img_c;
}

}   //namespace opendip