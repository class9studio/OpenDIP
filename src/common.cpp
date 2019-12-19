
#include <stdlib.h>
#include <iostream>
#include "common.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


namespace opendip{

/*****************************************************************************
*   Function name: ReadImage
*   Description  : read image in local filesystem(jpg, jpeg, bmp, png, raw...)
*   Parameters   : file_name            Image name
*                  p_image_data         Image data in mem after read
*   Return Value : negtive,0,positive, Error codes: negtive.
*   Spec         :
*   History:
*
*       1.  Date         : 2019-12-03
*           Author       : kingLCH
*           Modification : Created function
*****************************************************************************/
int ReadImage(char* file_name, unsigned char* p_image_data, long int image_size)
{
    if(file_name == NULL || p_image_data == NULL)
    {
        std::cout << "[common] Parameter invalid." << std::endl;
        return -1;
    }
    return 0;
}

/*****************************************************************************
*   Function name: WriteImage
*   Description  : write image from mem to local filesystem(jpg, jpeg, bmp, png, raw...)
*   Parameters   : file_name            Image name           
*                  p_image_data         Image data in mem
*   Return Value : negtive,0,positive, Error codes: negtive.
*   Spec         : 
*   History:
* 
*       1.  Date         : 2019-12-03
*           Author       : kingLCH
*           Modification : Created function
*****************************************************************************/
int WriteImage(char* file_name, unsigned char* p_image_data, long int image_size)
{
    if(file_name == NULL || p_image_data == NULL)
    {
        std::cout << "[common] Parameter invalid." << std::endl;
        return -1;
    }
    return 0;

}

/*****************************************************************************
*   Function name: ImgRead
*   Description  : read image in local filesystem(jpg, jpeg, bmp, png, raw...)
*   Parameters   : file_name            Image name
*   Return Value : class Image.
*   Spec         : image type:rgb
*   History:
*
*       1.  Date         : 2019-12-19
*           Author       : kingLCH
*           Modification : Created function
*****************************************************************************/
Image ImgRead(char* file_name)
{
	Image img;
	unsigned char* data;

	data = stbi_load(file_name, &img.w, &img.h, &img.c, 0);
	if (data == NULL)
	{
		printf("image load fail\n");
		return img;
	}

	img.cstep = img.w;
	img.ftype = GetImageTypeFromFile(file_name);
	img.data =(unsigned char*) data;

	return img;
}

/*****************************************************************************
*   Function name: ImgWrite
*   Description  : write image to local filesystem(jpg, jpeg, bmp, png, raw...)
*   Parameters   : file_name            Image name
*   Return Value : 0:write file success
*				   negtive:write file fail
*   Spec         : 
*   History:
*
*       1.  Date         : 2019-12-19
*           Author       : kingLCH
*           Modification : function draft
*****************************************************************************/
int ImgWrite(char* file_name, Image &img)
{
	int ret = 0;
	OpenDIP_Image_FILE_Type_e type = img.ftype;

	switch (type)
	{
	case OPENDIP_IMAGE_JPG:
		stbi_write_jpg(file_name, img.w, img.h, img.c, img.data, img.w * img.c);
		break;
	case OPENDIP_IMAGE_PNG:
		stbi_write_png(file_name, img.w, img.h, img.c, img.data, img.w * img.c);
		break;
	default:
		ret = -1;
		break;
	}
	return ret;
}

/*****************************************************************************
*   Function name: GetImageTypeFromFile
*   Description  : get image file type(jpg, jpeg, bmp, png, raw...)
*   Parameters   : file path
*   Return Value : OpenDIP_Image_FILE_Type_e
*   Spec         :
*   History:
*
*       1.  Date         : 2019-12-19
*           Author       : kingLCH
*           Modification : Created function
*****************************************************************************/
OpenDIP_Image_FILE_Type_e GetImageTypeFromFile(char *filename)
{
	OpenDIP_Image_FILE_Type_e  image_type = OPENDIP_IMAGE_UNKOWN;
	unsigned char file_size = 0;
	unsigned char index = 0;
	char suffix[64] = { 0 };
	if (filename == NULL)
	{
		std::cout << "filename not exist." << std::endl;
		return image_type;
	}
	file_size = strlen(filename);
	index = file_size;
	while ('.' != filename[index - 1] && index >= 0)
	{
		index--;
	}
	strcpy(suffix, (char *)(filename + index));
	printf("suffix: %s\n", suffix);

	if (0 == strcmp(suffix, "raw"))
	{
		image_type = OPENDIP_IMAGE_RAW;
	}
	else if (0 == strcmp(suffix, "jpg"))
	{
		image_type = OPENDIP_IMAGE_JPG;
	}
	else if (0 == strcmp(suffix, "tif"))
	{
		image_type = OPENDIP_IMAGE_TIF;
	}
	else if (0 == strcmp(suffix, "png"))
	{
		image_type = OPENDIP_IMAGE_PNG;
	}
	else if (0 == strcmp(suffix, "bmp"))
	{
		image_type = OPENDIP_IMAGE_BMP;
	}
	else if (0 == strcmp(suffix, "gip"))
	{
		image_type = OPENDIP_IMAGE_GIP;
	}
	else if (0 == strcmp(suffix, "ico"))
	{
		image_type = OPENDIP_IMAGE_ICO;
	}
	else
	{
		image_type = OPENDIP_IMAGE_UNKOWN;
	}

	return image_type;
}


}  //namespace opendip