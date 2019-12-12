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
#include <stdlib.h>
#include <iostream>

#include "common.h"

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
