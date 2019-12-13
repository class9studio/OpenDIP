#ifndef __IMAGE_H__
#define __IMAGE_H__

typedef enum OpenDIP_Image_Type_s
{
    OPENDIP_IMAGE_UNKOWN = 0x0,
    OPENDIP_IMAGE_RAW,
    OPENDIP_IMAGE_JPG,
    OPENDIP_IMAGE_TIF,
    OPENDIP_IMAGE_PNG,
    OPENDIP_IMAGE_BMP,
    OPENDIP_IMAGE_GIP,
    OPENDIP_IMAGE_ICO,
    OPENDIP_IMAGE_NUM,
}OpenDIP_Image_Type_e;

extern OpenDIP_Image_Type_e GetImageTypeFromFile(char *filename);

#endif