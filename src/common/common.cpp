#include <stdlib.h>
#include <iostream>
#include "common.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


namespace opendip{

void ShowDebugInfo()
{
    std::cout<< "[File]: " << __FILE__ << std::endl;
	std::cout<< "[Line]: " << __LINE__ << std::endl;
	std::cout<< "[Function]: " << __FUNCTION__<< std::endl;
}
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

	img.is_stbimage = true;
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
	img.ftype = GetImageTypeFromFile(file_name);

	switch (img.ftype)
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

/*****************************************************************************
*   Function name: StbFree
*   Description  : free stb-image data space
*   Parameters   : ptr           pointer to free
*   Return Value : void
*   Spec         : 
*   History:
*
*       1.  Date         : 2019-12-23
*           Author       : YangLin
*           Modification : function draft
*****************************************************************************/
void StbFree(void* ptr)
{
    if (ptr)
    {
       stbi_image_free(ptr);
    }
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
	if(src.data == NULL || src.w <= 1 || src.h <= 1 || src.h < channel)
	{
		Image res_img;
		return res_img;
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
		vector<Image> res_img;
		return res_img;
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

}  //namespace opendip