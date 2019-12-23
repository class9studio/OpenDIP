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

#include "image.h"

namespace opendip {
	void ShowDebugInfo();

	//read image data
	int ReadImage(char* file_name, unsigned char* p_image_data, long int image_size);

	//write image
	int WriteImage(char* file_name, unsigned char* p_image_data, long int image_size);

	//read image and return Image class
	Image ImgRead(char* file_name);

	//read image and return Image class
	int ImgWrite(char* file_name, Image &img);

	//get image file type
	OpenDIP_Image_FILE_Type_e GetImageTypeFromFile(char *filename);

	//free stb-image api alloc space
	void StbFree(void* ptr);
}
#endif
