/*///////////////////////////////////////////////////////////////////////////////////////
//
//                           License Agreement
//                For Open source Digital Image Processing Library(OpenDIP)
//
////////////////////////////////////////////////////////////////////////////////////////
//                    This is a base function head file.
//
//  File Name     : common.h
//  Version       : Initial Draft
//  Author        : KinglCH
//  Created       : 2019/12/04
//  Description   : 
//  1.Date        : 2019/12/04
//    Modification: Created file
//
///////////////////////////////////////////////////////////////////////////////////////*/

#ifndef _OPENDIP_COMMON_H_
#define _OPENDIP_COMMON_H_
#include <vector>
#include <string>
#include <Eigen/Dense>

#include "image.h"
#include "point.h"
using namespace std;
using namespace Eigen;

#define OPENDIP_PI   3.1415926535897932384626433832795
namespace opendip
{
enum Thresh_Binary_Type
{
	THRESH_BINARY = 0x0,
	THRESH_BINARY_INV,
	THRESH_TRUNC,
	THRESH_TOZERO,
	THRESH_TOZERO_INV,
};

void ShowDebugInfo();

//read image data
int ReadImage(char *file_name, unsigned char *p_image_data, long int image_size);

//write image
int WriteImage(char *file_name, unsigned char *p_image_data, long int image_size);

//read image and return Image class
Image ImgRead(string file_name);

//read image and return Image class
int ImgWrite(string file_name, Image &img);

//get image file type
OpenDIP_Image_FILE_Type GetImageTypeFromFile(const char *filename);

//free stb-image api alloc space
void StbFree(void *ptr);

// sperate one channel from image
Image Split(Image &src, OpenDIP_Channel_Type channel);

// sperate channels from image
vector<Image> Split(Image &src);

// merge channels to image
Image Merge(vector<Image> &channels, int num);

// color to grayscale conversion
Image ColorCvtGray(Image &src, OpenDIP_ColorCvtGray_Type cvt_type);

// mean and stddev in one channel image
void MeanStddev(Image &src, double *mean, double *stddev);

// max and min gray in one channel image
void MinMaxLoc(Image &src, unsigned char *min, unsigned char *max, Point &min_loc, Point &max_loc);

// image convert to Mat format
typedef Matrix<unsigned char, Dynamic, Dynamic, RowMajor> RowMatrixXc;
typedef Map<RowMatrixXc> MapType;
typedef Map<const RowMatrixXc> MapTypeConst; // a read-only map
											 // 单通道图像数据映射到Map中
MapType ImageCvtMap(Image &src);
MapTypeConst ImageCvtMapConst(Image &src);

// Ostu计算阈值
unsigned char GetOstu(Image &src);
// Image Binarization
Image Threshold(Image &src, Thresh_Binary_Type type, double threshold, double max_value, bool auto_threshold);

// 通过旋转角度和旋转中心，返回图像旋转矩阵2x3
Matrix<double, 2, 3> GetRotationMatrix2D(Point2f center, double angle, double scale);

// 仿射变换
Image WarpAffine(Image &src, Matrix<double, 2, 3> transform);

//彩色图像的直方图均衡
Image HistEqualization(Image &src);

}; // namespace opendip
#endif
