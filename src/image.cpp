#include <stdlib.h>
#include <iostream>
#include<cstring>
#include "image.h"

OpenDIP_Image_Type_e GetImageTypeFromFile(char *filename)
{
    OpenDIP_Image_Type_e  image_type = OPENDIP_IMAGE_UNKOWN; 
    unsigned char file_size = 0;
    unsigned char index = 0;
    char suffix[64] = {0};
    if(filename == NULL)
    {
        std::cout << "filename not exist." << std::endl;
        return image_type;
    }
    file_size = strlen(filename);
    index = file_size;
    while('.'!=filename[index-1] && index>=0)
    {
        index--;
    }
    strcpy(suffix, (char *)(filename+index));
    //printf("suffix: %s\n", suffix);

    if(0 == strcmp(suffix, "raw"))
    {
        image_type = OPENDIP_IMAGE_RAW;
    }
    else if(0 == strcmp(suffix, "jpg"))
    {
        image_type = OPENDIP_IMAGE_JPG;
    }
    else if(0 == strcmp(suffix, "tif"))
    {
        image_type = OPENDIP_IMAGE_TIF;
    }
    else if(0 == strcmp(suffix, "png"))
    {
        image_type = OPENDIP_IMAGE_PNG;
    }
    else if(0 == strcmp(suffix, "bmp"))
    {
        image_type = OPENDIP_IMAGE_BMP;
    }
    else if(0 == strcmp(suffix, "gip"))
    {
        image_type = OPENDIP_IMAGE_GIP;
    }
    else if(0 == strcmp(suffix, "ico"))
    {
        image_type = OPENDIP_IMAGE_ICO;
    }
    else
    {
        image_type = OPENDIP_IMAGE_UNKOWN;
    }

    return image_type;
}