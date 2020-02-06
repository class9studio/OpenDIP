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
}