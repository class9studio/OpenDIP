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
*   Function name: ConnectedComponents
*   Description  : 提取图像中不同连通域
*   Parameters   : src			   输入原始图像     
*   Return Value : int             连通域的个数                     
*   Spec         : 
*   History:
*
*       1.  Date         : 2020-1-20
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
int ConnectedComponents(Image &src, Image &labels)
{
	if(src.data == NULL || src.w < 1 || src.h < 1 || src.c != 1)
	{
		cout << "source image invalid" << endl;
		return -1;
	}

	int pix_val = 0;
	int label_seq = 1;
	int label_dst = 0;
	int pos_left = 0, pos_top = 0;
	int lab_left = 0, lab_top = 0;
	int lab_min = 0, lab_max = 0;
	map<int,int> labs_map;
	map<int,int>::iterator lr_labs;
	unsigned char *p_src_data = (unsigned char *)src.data;
	unsigned char *p_lab_data = (unsigned char *)labels.data;
	memset(p_lab_data, 0, src.w*src.h*src.c);
	//第一次遍历
	for(int j = 0; j < src.h; j++)
	{
		for(int i = 0; i < src.w; i++)
		{
			pix_val = p_src_data[j*src.w*src.c + i*src.c];
			if(pix_val == 0)
			{
				//背景点直接跳过
				continue;
			}
			else
			{
				pos_left = i - 1; //左边像素点
				pos_top = j - 1;   //上面像素点
				if(pos_left < 0 && pos_top < 0)
				{
					//图像左上角像素点
					p_lab_data[j*src.w*src.c + i*src.c] = label_seq; //第一个标签
					label_seq++;
				}
				else if(pos_left < 0 && pos_top >= 0)
				{
					//左边像素点缺失， 上面像素点存在
					lab_top = p_lab_data[(j-1)*src.w*src.c+ i*src.c];
					if(0 == lab_top)
					{
						p_lab_data[j*src.w*src.c + i*src.c] = label_seq;
						label_seq++;
					}
				}
				else if(pos_left >= 0 && pos_top < 0)
				{
					//左边像素点存在 上面像素点缺失
					lab_left = p_lab_data[j*src.w*src.c + (i-1)*src.c];
					if(0 == lab_left)
					{
						p_lab_data[j*src.w*src.c + i*src.c] = label_seq;
						label_seq++;
					}
				}
				else
				{
					/* 左像素、上像素均存在 */
					lab_left = p_lab_data[j*src.w*src.c + (i-1)*src.c];
					lab_top = p_lab_data[(j-1)*src.w*src.c + i*src.c];
					if(lab_left == 0 && lab_top == 0)
					{
						p_lab_data[j*src.w*src.c + i*src.c] = label_seq;
						label_seq++;
					}
					else if(lab_left == lab_top)
					{
						p_lab_data[j*src.w*src.c + i*src.c] = lab_left;
					}
					else
					{
						lab_min = (lab_left<lab_top)?lab_left:lab_top;
						lab_max = (lab_left>lab_top)?lab_left:lab_top;
						p_lab_data[j*src.w*src.c + i*src.c] = lab_min;
						//大label是小label的子域
						labs_map.insert(pair<int,int>(lab_max,lab_min));
					}	
				}
			}
		}
	}

	//第二次遍历
	for(int j = 0; j < src.h; j++)
	{
		for(int i = 0; i < src.w; i++)
		{
			//跳过背景点
			label_dst = p_lab_data[j*src.w*src.c + i*src.c];
			if(0 == label_dst)
			{
				continue;
			}
			//判断有无继承
			lr_labs = labs_map.find(label_dst);
			if(lr_labs != labs_map.end())
			{
				p_lab_data[j*src.w*src.c + i*src.c] = lr_labs->second;
			}
		}
	}

	return (label_seq - labs_map.size());
} 

/*****************************************************************************
*   Function name: GetStructuringElement
*   Description  : 生成常用的矩形结构元素、十字结构元素
*   Parameters   : shape			   结构元素形状  0-矩形  1-十字    
*                : ksize               连通域的个数                     
*   Return Value : MatrixXd            返回矩阵
*   Spec         : 
*   History:
*
*       1.  Date         : 2020-2-6
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
MatrixXd GetStructuringElement(int shape, int ksize)
{
	assert(ksize % 2 == 1 || shape == 0 || shape == 1);
	MatrixXd m(ksize, ksize);
	if(0 == shape)
	{
		// MORPH_RECT, 矩形结构元素，所有元素都为1
		m = MatrixXd::Ones(ksize, ksize);
	}
	else
	{
		// MORPH_CROSS, 十字结构元素，中间的列和行元素为1
		m = MatrixXd::Zero(ksize, ksize);
		VectorXd vec = VectorXd::Ones(ksize);
		m.row(ksize/2) = vec;
		m.col(ksize/2) = vec;
	}
	
	return m;
}

/*****************************************************************************
*   Function name: MorphErode
*   Description  : 图像腐蚀
*   Parameters   : src			       输入的待腐蚀图像    
*                : kernel              用于腐蚀操作的结构元素                     
*   Return Value : Image               腐蚀图像输出图像
*   Spec         : 
*   History:
*
*       1.  Date         : 2020-2-6
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image MorphErode(Image &src, MatrixXd kernel, int padding)
{
	assert(src.data != NULL  || src.c == 1 || src.c == 3 || src.w > 1 || src.h > 1);
	assert(kernel.rows()==kernel.cols() || kernel.rows()%2 == 1);

	int row = kernel.rows();
	int col = kernel.cols();
	int value = 0;
	bool flag = true;
	if(padding == 0)
		padding = kernel.rows() / 2;

	Image dst(src.w+padding*2-col+1, src.h+padding*2-row+1 , src.c);
	Image dst_bound(src.w + padding*2, src.h + padding*2, src.c);
	unsigned char *p_src_data = (unsigned char*)src.data;
	unsigned char *p_dst_data = (unsigned char*)dst.data;
	unsigned char *p_dst_bound_data = (unsigned char*)dst_bound.data;

	memset(p_dst_data, 0, dst.w*dst.h*dst.c);
	// 腐蚀，边界填充1
	memset(p_dst_bound_data, 1, dst_bound.w*dst_bound.h*dst_bound.c);	
	//填充操作
	for(int j = 0; j < src.h; j++)
	{
		for(int i = 0; i < src.w; i++)
		{
			for(int z = 0; z < src.c; z++)
			{
				p_dst_bound_data[(j+padding)*dst_bound.c*dst_bound.w + (i+padding)*dst_bound.c + z] = p_src_data[j*src.c*src.w + i*src.c + z];
			}
		}
	}

	//扫描矩阵
	int m = 0, n = 0;
	unsigned char pixel_val = 0;

	for(int j = 0; j < dst_bound.h - row + 1; j++)
	{
		for(int i = 0; i < dst_bound.w - col + 1; i++)
		{
			for(int z = 0; z < dst_bound.c; z++)
			{
				pixel_val = p_dst_bound_data[(j+row/2)*dst_bound.c*dst_bound.w + (i+col/2)*dst_bound.c + z];
				
				if(pixel_val != 0)  //只对非0像素值处理
				{
					flag = true;
					for(m = j; m < j + row; m++)
					{
						for(n = i;n < i + col; n++)
						{
							if(kernel(m-j,n-i) == 1 && 0 == p_dst_bound_data[m*dst_bound.c*dst_bound.w + n*dst_bound.c + z])
							{
								flag = false;
								break;
							}
						}
					}

					if(true == flag)
						p_dst_data[j*dst.c*dst.w + i*dst.c + z] = pixel_val;
	
				}
			}
		}
	}

	return dst;
}

/*****************************************************************************
*   Function name: MorphDilate
*   Description  : 图像膨胀
*   Parameters   : src			       输入的待膨胀图像    
*                : kernel              用于腐蚀操作的结构元素                     
*   Return Value : Image               膨胀图像输出图像
*   Spec         : 
*   History:
*
*       1.  Date         : 2020-2-6
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image MorphDilate(Image &src, MatrixXd kernel, int padding)
{
	assert(src.data != NULL  || src.c == 1 || src.c == 3 || src.w > 1 || src.h > 1);
	assert(kernel.rows()==kernel.cols() || kernel.rows()%2 == 1);

	int row = kernel.rows();
	int col = kernel.cols();
	int value = 0;
	bool flag = true;
	if(padding == 0)
		padding = kernel.rows() / 2;

	Image dst(src.w+padding*2-col+1, src.h+padding*2-row+1 , src.c);
	Image dst_bound(src.w + padding*2, src.h + padding*2, src.c);
	unsigned char *p_src_data = (unsigned char*)src.data;
	unsigned char *p_dst_data = (unsigned char*)dst.data;
	unsigned char *p_dst_bound_data = (unsigned char*)dst_bound.data;

	memset(p_dst_data, 0, dst.w*dst.h*dst.c);
	memset(p_dst_bound_data, 1, dst_bound.w*dst_bound.h*dst_bound.c);	
	//填充操作
	for(int j = 0; j < src.h; j++)
	{
		for(int i = 0; i < src.w; i++)
		{
			for(int z = 0; z < src.c; z++)
			{
				p_dst_bound_data[(j+padding)*dst_bound.c*dst_bound.w + (i+padding)*dst_bound.c + z] = p_src_data[j*src.c*src.w + i*src.c + z];
			}
		}
	}

	//扫描矩阵
	int m = 0, n = 0;
	int pixel_val = 0;

	for(int j = 0; j < dst_bound.h - row + 1; j++)
	{
		for(int i = 0; i < dst_bound.w - col + 1; i++)
		{
			for(int z = 0; z < dst_bound.c; z++)
			{
				pixel_val = p_dst_bound_data[(j+row/2)*dst_bound.c*dst_bound.w + (i+col/2)*dst_bound.c + z];
				
				if(pixel_val != 0)  //只对非0像素值处理
				{
					flag = true;
					for(m = j; m < j + row; m++)
					{
						for(n = i;n < i + col; n++)
						{
							if(kernel(m-j,n-i) == 1 && 0 == p_dst_bound_data[m*dst_bound.c*dst_bound.w + n*dst_bound.c + z])
							{
								flag = false;
								break;
							}
						}
					}
					if(false == flag)
					{
						for(int k = j-col/2 ; k < j+col/2 + 1; k++)
						{
							for(int l = i-row/2; l < i+row/2 + 1; l++)
							{
								if(k>=0 && l>=0 && 1==kernel(k-j+col/2,l-i+row/2))
								{
									p_dst_data[k*dst.c*dst.w + l*dst.c + z] = pixel_val;
								}
							}
						}
					}
					else
					{
						p_dst_data[j*dst.c*dst.w + i*dst.c + z] = pixel_val;
					}
				}
			}
		}
	}

	return dst;
}

/*****************************************************************************
*   Function name: MorphOpen
*   Description  : 形态学开运算
*   Parameters   : src			       输入的待运算图像    
*                : kernel              用于开操作的结构元素                     
*   Return Value : Image               开运算后图像输出图像
*   Spec         : 
*        开运算是图像腐蚀和膨胀操作的结合，首先对图像进行腐蚀，消除图像中的噪声和较小的连通域，之后通过膨胀运算弥补较大连通域因腐蚀而造成的面积减小
*   作用:去除图像中的噪声，消除较小连通域，保留较大连通域，同时能够在两个物体纤细的连接处将两个物体分离，并且在不明显改变较大连通域的面积的同时能够
*        平滑连通域的边界
*   History:
*
*       1.  Date         : 2020-2-8
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image MorphOpen(Image &src, MatrixXd kernel)
{
	Image dst = MorphErode(src, kernel);
	return MorphDilate(dst, kernel);
}

/*****************************************************************************
*   Function name: MorphClose
*   Description  : 形态学关运算
*   Parameters   : src			       输入的待运算图像    
*                : kernel              用于开操作的结构元素                     
*   Return Value : Image               关运算后图像输出图像
*   Spec         : 
*        开运算是图像腐蚀和膨胀操作的结合，膨胀+腐蚀
*   作用:  去除连通域内的小型空洞，平滑物体轮廓，连接两个临近的连通域
*   History:
*
*       1.  Date         : 2020-2-8
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image MorphClose(Image &src, MatrixXd kernel)
{
	Image dst = MorphDilate(src, kernel);
	return MorphErode(dst, kernel);
}

/*****************************************************************************
*   Function name: MorphGradient
*   Description  : 形态学梯度
*   Parameters   : src			       输入的待运算图像    
*                : kernel              用于梯度的结构元素 
*                : type                梯度类型: 基本、内部和外部                 
*   Return Value : Image               梯度后图像输出图像
*   Spec         : 
*        基本梯度是原图像膨胀后图像和腐蚀后图像间的差值图像
*        内部梯度图像是原图像和腐蚀后图像间的差值图像
*        外部梯度是膨胀后图像和原图像间的差值图像   
*   History:
*
*       1.  Date         : 2020-2-8
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image MorphGradient(Image &src, MatrixXd kernel, Morph_Gradient_Type type)
{
	assert(type <= MORPH_GRADIENT_OUTSIDE);

	MapType src_m = ImageCvtMap(src);
	switch(type)
	{
		case MORPH_GRADIENT_BASIC:
		{
			Image dst_dilate = MorphDilate(src, kernel);
			Image dst_erode  = MorphErode(src, kernel);
			MapType dst_dilate_m = ImageCvtMap(dst_dilate);
			MapType dst_erode_m  = ImageCvtMap(dst_erode);
			dst_dilate_m -= dst_erode_m;
			return dst_dilate;
		}
		case MORPH_GRADIENT_INSIDE:
		{
			Image dst_erode = MorphErode(src, kernel);
			MapType dst_erode_m = ImageCvtMap(dst_erode);
			dst_erode_m -= src_m;
			return dst_erode;
		}
		case MORPH_GRADIENT_OUTSIDE:
		{
			Image dst_dilate = MorphDilate(src, kernel);
			MapType dst_dilate_m = ImageCvtMap(dst_dilate);
			dst_dilate_m -= src_m;
			return dst_dilate;
		}
		default:
			assert(false);
			break;
	}
}

//形态学梯度，轮廓发现运用

/*****************************************************************************
*   Function name: MorphTophat
*   Description  : 顶帽操作
*   Parameters   : src			       输入的待运算图像    
*                : kernel              用于顶帽操作的结构元素                     
*   Return Value : Image               顶帽操作后图像输出图像
*   Spec         : 
*   History:
*       图像顶帽运算是原图像与开运算结果之间的差值，往往用来分离比邻近点亮一些的斑块
*
*       1.  Date         : 2020-2-8
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image MorphTophat(Image &src, MatrixXd kernel)
{
	Image dst_open = MorphOpen(src, kernel);
	MapType src_m = ImageCvtMap(src);
	MapType dst_open_m = ImageCvtMap(dst_open);
	dst_open_m -= src_m;
	return dst_open;
}

/*****************************************************************************
*   Function name: MorphBlackhat
*   Description  : 黑帽操作
*   Parameters   : src			       输入的待运算图像    
*                : kernel              用于黑帽操作的结构元素                     
*   Return Value : Image               黑帽操作后图像输出图像
*   Spec         : 
*   History:
*       黑帽运算是原图像与闭运算结果之间的差值，往往用来分离比邻近点暗一些的斑块
*
*       1.  Date         : 2020-2-8
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
Image MorphBlackhat(Image &src, MatrixXd kernel)
{
	Image dst_close = MorphClose(src, kernel);
	MapType src_m = ImageCvtMap(src);
	MapType dst_close_m = ImageCvtMap(dst_close);
	dst_close_m -= src_m;
	return dst_close;
}

}