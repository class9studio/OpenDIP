#include <iostream>
#include <cmath>

#include "common.h"
#include "algorithm.h"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace opendip {
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
*   Function name: GetOstu
*   Description  : OSTU（大津算法）
*   Parameters   : src              source image 
*
*   Return Value : 阈值灰度
*   Spec         : 
*   History:
*
*       1.  Date         : 2019-12-30
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
unsigned char GetOstu(Image &src)
{
	unsigned int *histogram = new unsigned int[256]();
	unsigned char *p_src_data = (unsigned char *)src.data;  
	for(size_t j = 0; j < src.h; j++)
    {
        for(size_t i = 0; i < src.w; i++)
        {
			histogram[p_src_data[j * src.c * src.w + src.c*i]] += 1;
        }
    }
	long size = src.w * src.h;
	unsigned char threshold;      
	long sum0 = 0, sum1 = 0; //存储前景的灰度总和及背景灰度总和  
	long cnt0 = 0, cnt1 = 0; //前景的总个数及背景的总个数  
	double w0 = 0, w1 = 0; //前景及背景所占整幅图像的比例  
	double u0 = 0, u1 = 0;  //前景及背景的平均灰度  
	double variance = 0; //最大类间方差
	double maxVariance = 0;  
	for(int i = 1; i < 256; i++) //一次遍历每个像素  
	{    
		sum0 = 0;  
		sum1 = 0;   
		cnt0 = 0;  
		cnt1 = 0;  
		w0 = 0;  
		w1 = 0;  
		for(int j = 0; j < i; j++)  
		{  
			cnt0 += histogram[j];  
			sum0 += j * histogram[j];  
		}  
 
		u0 = (double)sum0 /  cnt0;   
		w0 = (double)cnt0 / size;  
 
		for(int j = i ; j <= 255; j++)  
		{  
			cnt1 += histogram[j];  
			sum1 += j * histogram[j];  
		}  
 
		u1 = (double)sum1 / cnt1;  
		w1 = 1 - w0; // (double)cnt1 / size;  
 
		variance =  w0 * w1 *  (u0 - u1) * (u0 - u1);  
		if(variance > maxVariance)   
		{    
			maxVariance = variance;    
			threshold = i;    
		}   
	}    

	return threshold;
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

}