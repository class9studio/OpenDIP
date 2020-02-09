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

namespace opendip
{
	#define OPENDIP_PI   3.1415926535897932384626433832795
	// image convert to Mat format
    typedef Matrix<unsigned char, Dynamic, Dynamic, RowMajor> RowMatrixXc;
    typedef Map<RowMatrixXc, Unaligned, InnerStride<1>> GrayImgMap;
    typedef Map<const RowMatrixXc, Unaligned, InnerStride<1>> GrayImgMapConst;

    typedef Map<RowMatrixXc, Unaligned, InnerStride<3>> ColorImgMap;
    typedef Map<const RowMatrixXc, Unaligned, InnerStride<3>> ColorImgMapConst;

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

	enum Thresh_Binary_Type
	{
		THRESH_BINARY = 0x0,
		THRESH_BINARY_INV,
		THRESH_TRUNC,
		THRESH_TOZERO,
		THRESH_TOZERO_INV,
	};

    enum Morph_Gradient_Type
	{
		MORPH_GRADIENT_BASIC = 0x0,
		MORPH_GRADIENT_INSIDE,
		MORPH_GRADIENT_OUTSIDE,
	};
}; // namespace opendip
#endif
