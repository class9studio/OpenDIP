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

#ifndef _COMMON_H_
#define _COMMON_H_

//read image data
int ReadImage(char* file_name, unsigned char* p_image_data);

//write image
int WriteImage(char* file_name, unsigned char* p_image_data);

#endif