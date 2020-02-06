#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <map>

#include "common.h"
#include "algorithm.h"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace opendip {
/*****************************************************************************
*   Function name: ColorCvtGray
*   Description  : color to grayscale conversion
*   Parameters   : src              source image
*    			   cvt_type         convert methods
*   Return Value : Image            dst image
*   Spec         : 
*   History:
*
*       1.  Date         : 2019-12-24
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image ColorCvtGray(Image &src, OpenDIP_ColorCvtGray_Type cvt_type)
{
	if(src.c != 3)
	{
		return Image();
	}

	Image img_cvt(src.w, src.h, 1);
	unsigned char* p_src_data =(unsigned char*) src.data;
	unsigned char* p_dst_data =(unsigned char*) img_cvt.data;
    for(size_t j = 0; j < src.h; j++)
    {
        for(size_t i = 0; i < src.w; i++)
        {
			unsigned char mix_gray = 0;
			unsigned char rValue = p_src_data[j * src.c * src.w + src.c*i + 0];
            unsigned char gValue = p_src_data[j * src.c * src.w + src.c*i + 1];
            unsigned char bValue = p_src_data[j * src.c * src.w + src.c*i + 2];
			switch(cvt_type)
			{
				case OPENDIP_COLORCVTGRAY_MAXMIN:
				{
					unsigned char maxValue;
					unsigned char minValue;
					
					maxValue = max(rValue,gValue);
					maxValue = max(bValue,maxValue);
					
					minValue = min(rValue,gValue);
					minValue = min(bValue,minValue);

					p_dst_data[j * img_cvt.c * img_cvt.w + img_cvt.c*i] = (unsigned char) ((maxValue + minValue ) / 2.0);
					break;
				}
				case OPENDIP_COLORCVTGRAY_AVERAGE:
					p_dst_data[j * img_cvt.c * img_cvt.w + img_cvt.c*i] = (unsigned char) ((rValue + gValue + bValue) / 3.0);
					break;
				case OPENDIP_COLORCVTGRAY_WEIGHTED:
					p_dst_data[j * img_cvt.c * img_cvt.w + img_cvt.c*i] = (unsigned char) (0.3 * rValue + 0.59 * gValue+ 0.11 * bValue);
					break;
				default:
					p_dst_data[j * img_cvt.c * img_cvt.w + img_cvt.c*i] = (unsigned char) (0.3 * rValue + 0.59 * gValue+ 0.11 * bValue);
					break; 				
			}
        }
    }

	return img_cvt;
}

/*****************************************************************************
*   Function name: Threshold
*   Description  : Image Binarization
*   Parameters   : src              image to Map
*                  type             binarization type
*                  threshold        二值化的阈值
*                  max_value        二值化过程中的最大值
*                  auto_threshold   自动阈值标志
*   Return Value : Map              Map of image
*   Spec         : 
*   History:
*
*       1.  Date         : 2019-12-30
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image Threshold(Image &src, Thresh_Binary_Type type, double threshold, double max_value, bool auto_threshold)
{
	if(auto_threshold)
	{
		// update threshold
		threshold =(unsigned char) GetOstu(src);
	}

	if(src.data == NULL || src.w <= 1 || src.h <= 1 || src.c != 1)
	{
		return Image();
	}

	Image img_dst(src.w, src.h, 1);
	unsigned char* p_src_data =(unsigned char*) src.data;
	unsigned char* p_dst_data =(unsigned char*) img_dst.data;
	unsigned char gray = 0;
    for(size_t j = 0; j < src.h; j++)
    {
        for(size_t i = 0; i < src.w; i++)
        {
			gray = p_src_data[j * src.c * src.w + src.c*i];
			switch(type)
			{
				case THRESH_BINARY:
					if(gray > threshold)
						p_dst_data[j * img_dst.c * img_dst.w + img_dst.c*i] = max_value;
					else
						p_dst_data[j * img_dst.c * img_dst.w + img_dst.c*i] = 0;
					break;
				case THRESH_BINARY_INV:
					if(gray > threshold)
						p_dst_data[j * img_dst.c * img_dst.w + img_dst.c*i] = 0;
					else
						p_dst_data[j * img_dst.c * img_dst.w + img_dst.c*i] = max_value;
					break;
				case THRESH_TRUNC:
					if(gray > threshold)
						p_dst_data[j * img_dst.c * img_dst.w + img_dst.c*i] = threshold;
					break;
				case THRESH_TOZERO:
					if(gray <= threshold)
						p_dst_data[j * img_dst.c * img_dst.w + img_dst.c*i] = max_value;
					break;
				case THRESH_TOZERO_INV:
					if(gray > threshold)
						p_dst_data[j * img_dst.c * img_dst.w + img_dst.c*i] = max_value;					
					break;
			}
        }
    }

	return img_dst;	
}

/*****************************************************************************
*   Function name: HistEqualazition
*   Description  : 彩色图像的直方图均衡
*                  如果一个图像的直方图都集中在一个区域，则整体图像的对比度比较小，不便于图像中纹理的识别
*                  将图像中灰度值的范围扩大，增加原来两个灰度值之间的差值，就可以提高图像的对比度，进而将图像中的纹理突出显现出来
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
Image HistEqualazition(Image &src)
{
	if(src.data == NULL || src.w <= 1 || src.h <= 1 || src.c != 3)
	{
		return Image();
	}
	Image dst_image(src.w, src.h, src.c);
	unsigned int *hist_r = new unsigned int [256]();
	unsigned int *hist_g = new unsigned int [256]();
	unsigned int *hist_b = new unsigned int [256]();
	
	float *hist_tmp_r = new float [256]();
    float *hist_tmp_g = new float [256]();
    float *hist_tmp_b = new float [256]();
	
	//Calculate Histograms of original image
	unsigned char *p_src_data = (unsigned char *)src.data;  
	unsigned char *p_dst_data = (unsigned char *)dst_image.data;  
	for(size_t j = 0; j < src.h; j++)
    {
        for(size_t i = 0; i < src.w; i++)
        {
			hist_r[p_src_data[j * src.c * src.w + src.c*i + 0]] += 1;
			hist_g[p_src_data[j * src.c * src.w + src.c*i + 1]] += 1;
			hist_b[p_src_data[j * src.c * src.w + src.c*i + 2]] += 1;
        }
    }

    // Calculate normalized histogram
    for(size_t i = 0; i < 256; i++)
	{
        hist_tmp_r[i] = hist_r[i] / (((float)(src.h))*((float)(src.w)));
        hist_tmp_g[i] = hist_g[i] / (((float)(src.h))*((float)(src.w)));
        hist_tmp_b[i] = hist_b[i] / (((float)(src.h))*((float)(src.w)));
    }

    // Calculate cdf
    for(int i = 0; i < 256; i++) 
	{
        if(i == 0) {
            hist_tmp_r[i] = hist_tmp_r[i];
            hist_tmp_g[i] = hist_tmp_g[i];
            hist_tmp_g[i] = hist_tmp_b[i];
        }
        else{
            hist_tmp_r[i] = hist_tmp_r[i] + hist_tmp_r[i-1];
            hist_tmp_g[i] = hist_tmp_g[i] + hist_tmp_g[i-1];
            hist_tmp_g[i] = hist_tmp_g[i] + hist_tmp_g[i-1];
        }
    }

    float scalingFactor = 255.0;
    for(int i = 0; i < 256; i++) 
	{
		hist_tmp_r[i] = hist_tmp_r[i] * scalingFactor;
		hist_tmp_g[i] = hist_tmp_g[i] * scalingFactor;
		hist_tmp_g[i] = hist_tmp_b[i] * scalingFactor;
    }

    for(size_t j = 0; j < dst_image.h; j++)
    {
        for(size_t i = 0; i < dst_image.w; i++)
        {
			unsigned int rPixel = p_src_data[j * src.c * src.w + src.c*i + 0];
            p_dst_data[j * dst_image.c * dst_image.w + dst_image.c*i + 0] = (unsigned char)floor(hist_tmp_r[rPixel]);

			unsigned int gPixel = p_src_data[j * src.c * src.w + src.c*i + 1];
            p_dst_data[j * dst_image.c * dst_image.w + dst_image.c*i + 1] = (unsigned char)floor(hist_tmp_r[rPixel]);

			unsigned int bPixel = p_src_data[j * src.c * src.w + src.c*i + 2];
            p_dst_data[j * dst_image.c * dst_image.w + dst_image.c*i + 2] = (unsigned char)floor(hist_tmp_r[rPixel]);
        }
    }

	return dst_image;
}

/*****************************************************************************
*   Function name: HistEqualazition
*   Description  : 灰度图像的直方图均衡
*   Parameters   : src：			 输入灰度图像      
*                  transform         2×3的变换矩阵
*   Return Value : Image             输出图像
*   Spec         : 
*   History:
*
*       1.  Date         : 2020-1-14
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image HistEqualizationGray(Image &src)
{
	if(src.data == NULL || src.w <= 1 || src.h <= 1 || src.c != 1)
	{
		return Image();
	}
	Image dst_image(src.w, src.h, src.c);
	unsigned int *hist = new unsigned int [256]();
	float *hist_tmp = new float [256]();

	//Calculate Histograms of original image
	unsigned char *p_src_data = (unsigned char *)src.data;  
	unsigned char *p_dst_data = (unsigned char *)dst_image.data;  
	for(size_t j = 0; j < src.h; j++)
    {
        for(size_t i = 0; i < src.w; i++)
        {
			hist[p_src_data[j * src.c * src.w + src.c*i]] += 1;
        }
    }

    // Calculate normalized histogram
    for(size_t i = 0; i < 256; i++)
	{
        hist_tmp[i] = hist[i] / (((float)(src.h))*((float)(src.w)));
    }	

    // Calculate cdf
    for(int i = 0; i < 256; i++) 
	{
        if(i == 0) {
            hist_tmp[i] = hist_tmp[i];
        }
        else{
            hist_tmp[i] = hist_tmp[i] + hist_tmp[i-1];
        }
    }

    float scalingFactor = 255.0;			
    for(int i = 0; i < 256; i++) 
	{
		hist_tmp[i] = hist_tmp[i] * scalingFactor;
    }

    for(size_t j = 0; j < dst_image.h; j++)
    {
        for(size_t i = 0; i < dst_image.w; i++)
        {
			unsigned int rPixel = p_src_data[j * src.c * src.w + src.c*i];
            p_dst_data[j * dst_image.c * dst_image.w + dst_image.c*i] = (unsigned char)floor(hist_tmp[rPixel]);
        }
    }

	return dst_image;
}

/*****************************************************************************
*   Function name: HistEqualazition
*   Description  : 灰度图像的直方图配准
*   Parameters   : src1：			 输入灰度图像      
*                  src2:             配准目标图像
*   Return Value : Image             输出图像
*   Spec         : 
*   History:
*
*       1.  Date         : 2020-1-14
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image HistRegistration(Image &src1, Image &src2)
{
	if(src1.data == NULL || src1.w <= 1 || src1.h <= 1 || src1.c != 1 || src2.data == NULL || src2.c != 1)
	{
		return Image();
	}
	Image dst_image(src1.w, src1.h, src1.c);
	unsigned int *hist_src1 = new unsigned int [256]();
	float *hist_tmp_src1 = new float [256]();
	unsigned int *hist_src2 = new unsigned int [256]();
	float *hist_tmp_src2 = new float [256]();

	//Calculate Histograms of original image
	unsigned char *p_src1_data = (unsigned char *)src1.data;  
	unsigned char *p_src2_data = (unsigned char *)src2.data;  
	unsigned char *p_dst_data = (unsigned char *)dst_image.data;  
	for(size_t j = 0; j < src1.h; j++)
    {
        for(size_t i = 0; i < src1.w; i++)
        {
			hist_src1[p_src1_data[j * src1.c * src1.w + src1.c * i]] += 1;
        }
    }

	for(size_t j = 0; j < src2.h; j++)
	{
		for(size_t i = 0; i < src2.w; i++)
		{
			hist_src2[p_src2_data[j * src2.c * src2.w + src2.c * i]] += 1;
		}
	}

    // Calculate normalized histogram
    for(size_t i = 0; i < 256; i++)
	{
        hist_tmp_src1[i] = hist_src1[i] / (((float)(src1.h))*((float)(src1.w)));
        hist_tmp_src2[i] = hist_src2[i] / (((float)(src2.h))*((float)(src2.w)));
    }	

    // Calculate cdf
    for(int i = 0; i < 256; i++) 
	{
        if(i == 0) {
            hist_tmp_src1[i] = hist_tmp_src1[i];
            hist_tmp_src2[i] = hist_tmp_src2[i];
        }
        else{
            hist_tmp_src1[i] = hist_tmp_src1[i] + hist_tmp_src1[i-1];
            hist_tmp_src2[i] = hist_tmp_src2[i] + hist_tmp_src2[i-1];
        }
    }

	//构建累积概率误差矩阵
	float diff_cdf[256][256];
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			diff_cdf[i][j] = fabs(hist_tmp_src1[i] - hist_tmp_src2[j]);
		}
	}

	unsigned char *lut = new unsigned char[256]();
	unsigned char index = 0;
	for(int i = 0; i < 256; i++)
	{
		float min = diff_cdf[i][0];
		//寻找每一列中最小值
		for(size_t j = 0; j < 256; j++)
		{
			if(min > diff_cdf[i][j])
			{
				index = j;
				min = diff_cdf[i][j];
			}
		}
		lut[i] = index;
	}

    for(size_t j = 0; j < dst_image.h; j++)
    {
        for(size_t i = 0; i < dst_image.w; i++)
        {
			unsigned int rPixel = p_src1_data[j * src1.c * src1.w + src1.c*i];
            p_dst_data[j * dst_image.c * dst_image.w + dst_image.c*i] = (unsigned char)lut[rPixel];
        }
    }	
	return dst_image;
}

/*****************************************************************************
*   Function name: SaltAndPepper
*   Description  : 生成椒盐噪声
*   Parameters   : image：			 输出椒盐噪声尺寸(与原图像大小相同)      
*                  n:                噪声个数
*   Return Value : void              
*   Spec         : 
*       椒盐噪声又被称作脉冲噪声，它会随机改变图像中的像素值，是由相机成像、图像传输、解码处理等过程产生的黑白相间的亮暗点噪声，
*   其样子就像在图像上随机的撒上一些盐粒和黑椒粒，因此被称为椒盐噪声
*   History:
*
*       1.  Date         : 2020-1-15
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void SaltAndPepper(Image &src, int n)
{
    if(src.w < 1 || src.h < 1 || n < 0)
    {
        cout << "src image invalid" << endl;
        return;
    }
    unsigned char *p_src_data = (unsigned char *)src.data;
    for(int k = 0; k < n/2; k++)
    {
        int i = 0, j = 0;
        i = std::rand() % src.w;   //保证随机数不超出行列范围
        j = std::rand() % src.h;
        int is_white = std::rand() % 2;
        if(is_white)  //白噪声
        {
            if(src.c == 1)   //灰度图
            {
                p_src_data[j*src.c*src.w + i*src.c] = 255;
            }
            else if(src.c == 3)  //彩色图
            {
                p_src_data[j*src.c*src.w + i*src.c + 0] = 255;
                p_src_data[j*src.c*src.w + i*src.c + 1] = 255;
                p_src_data[j*src.c*src.w + i*src.c + 2] = 255;
            }
            else
            {
                cout << "Image size invalid when add white noise" << endl;
            }
        }
        else  //黑噪声
        {
            if(src.c == 1)   //灰度图
            {
                p_src_data[j*src.c*src.w + i*src.c] = 0;
            }
            else if(src.c == 3)  //彩色图
            {
                p_src_data[j*src.c*src.w + i*src.c + 0] = 0;
                p_src_data[j*src.c*src.w + i*src.c + 1] = 0;
                p_src_data[j*src.c*src.w + i*src.c + 2] = 0;
            }
            else
            {
                cout << "Image size invalid when add black noise" << endl;
            }
        }
    }
}

/*****************************************************************************
*   Function name: RandomGuassinGen
*   Description  : 生产高斯(正态)分布随机数
*   Parameters   : mean：			  正态分布均值
*                  sigma:             正态分布标准差     
*   Return Value : double             返回正态分布随机数            
*   History:
*
*       1.  Date         : 2020-1-15
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
double RandomGuassinGen(double mean, double sigma)    
{
    // construct a random generator engine from a time-based seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<double> dis(mean,sigma);
  
    return dis(gen);
}
/*****************************************************************************
*   Function name: GussianNoise
*   Description  : 高斯噪声函数-灰度图像
*   Parameters   : image：			 输出高斯噪声图像(与原图像大小相同)      
*   Return Value : void              
*   Spec         :   高斯噪声可能出现在图像的所有位置，所以需要用叠加的方式产生新图像
*                    图像的叠加使用Engin的Map映射成矩阵 直接进行矩阵加和
*   History:
*
*       1.  Date         : 2020-1-15
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void GussianNoiseImg_Gray(Image &src, double mean, double sigma)
{
    if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 0)
    {
        cout << "src image invalid" << endl;
        return;
    }
    //产生噪声图像
    Image gussain_noise(src.w, src.h, src.c);
    unsigned char *p_gus_noise = (unsigned char *)gussain_noise.data;
    for(int j = 0; j < gussain_noise.h; j++)
    {   
        for(int i = 0; i < gussain_noise.w; i++)
        {
            for(int z = 0; z < src.c; z++)
            {
                p_gus_noise[j*gussain_noise.w*gussain_noise.c + i*gussain_noise.c] = (unsigned char)RandomGuassinGen(mean, sigma);
            }
        }
    }

    //将image映射成Map的matrix
    MapType src_m1 = ImageCvtMap(src);
    MapType src_m2 = ImageCvtMap(gussain_noise);
    // map的加和操作会修改data的数据
    src_m1 = src_m1 + src_m2;
}

/*****************************************************************************
*   Function name: GussianNoise
*   Description  : 高斯噪声函数-自动识别通道
*   Parameters   : image：			 输出高斯噪声图像(与原图像大小相同)      
*   Return Value : void              
*   Spec         : 
*   History:
*
*       1.  Date         : 2020-1-15
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void GussianNoiseImg(Image &src, double mean, double sigma)
{
    if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 0)
    {
        cout << "src image invalid" << endl;
        return;
    }
	unsigned char rng = 0;
    unsigned char *p_src_data = (unsigned char *)src.data;
    for(int j = 0; j < src.h; j++)
    {   
        for(int i = 0; i < src.w; i++)
        {
			rng = (unsigned char)RandomGuassinGen(mean, sigma);
            for(int z = 0; z < src.c; z++)
            {
                p_src_data[j*src.w*src.c + i*src.c] += rng;
            }
        }
    }
}

}