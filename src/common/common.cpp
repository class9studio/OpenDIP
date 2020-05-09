#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "common.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


namespace opendip{

void ShowDebugInfo(void)
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
Image ImgRead(string file_name)
{
	Image img;
	unsigned char* data;

	data = stbi_load(file_name.c_str(), &img.w, &img.h, &img.c, 0);
	if (data == NULL)
	{
		printf("image load fail\n");
		return img;
	}

	img.cstep = img.w;
	img.ftype = GetImageTypeFromFile(file_name.c_str());

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
int ImgWrite(string file_name, Image &img)
{
	int ret = 0;
	img.ftype = GetImageTypeFromFile(file_name.c_str());

	switch (img.ftype)
	{
	case OPENDIP_IMAGE_JPG:
		stbi_write_jpg(file_name.c_str(), img.w, img.h, img.c, img.data, img.w * img.c);
		break;
	case OPENDIP_IMAGE_PNG:
		stbi_write_png(file_name.c_str(), img.w, img.h, img.c, img.data, img.w * img.c);
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
OpenDIP_Image_FILE_Type GetImageTypeFromFile(const char *filename)
{
	OpenDIP_Image_FILE_Type  image_type = OPENDIP_IMAGE_UNKOWN;
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


#include "common.h"

void launch_dummmy_kernel()
{

}

void print_array(int * input, const int array_size)
{
	for (int i = 0; i < array_size; i++)
	{
		if (!(i == (array_size - 1)))
		{
			printf("%d,", input[i]);
		}
		else
		{
			printf("%d \n", input[i]);
		}
	}
}

void print_array(float * input, const int array_size)
{
	for (int i = 0; i < array_size; i++)
	{
		if (!(i == (array_size - 1)))
		{
			printf("%f,", input[i]);
		}
		else
		{
			printf("%f \n", input[i]);
		}
	}
}

void print_matrix(int * matrix, int nx, int ny)
{
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%d ",matrix[nx * iy + ix]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_matrix(float * matrix, int nx, int ny)
{
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%.2f ", matrix[nx * iy + ix]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_arrays_toafile_side_by_side(float*a, float*b, int size, char* name)
{
	std::ofstream file(name);

	if (file.is_open())
	{
		for (int i = 0; i < size; i++) {
			file << i << " - " <<a[i] << " - " << b[i] << "\n";
		}
		file.close();
	}
}

void print_arrays_toafile_side_by_side(int*a, int*b, int size, char* name)
{
	std::ofstream file(name);

	if (file.is_open())
	{
		for (int i = 0; i < size; i++) {
			file << i << " - " << a[i] << " - " << b[i] << "\n";
		}
		file.close();
	}
}

void print_arrays_toafile(int*a, int size, char* name)
{
	std::ofstream file(name);

	if (file.is_open())
	{
		for (int i = 0; i < size; i++) {
			file << i << " - " << a[i] << "\n";
		}
		file.close();
	}
}

int* get_matrix(int rows, int columns)
{
	int mat_size = rows * columns;
	int mat_byte_size = sizeof(int)*mat_size;

	int * mat = (int*)malloc(mat_byte_size);

	for (int i = 0; i < mat_size; i++)
	{
		if (i % 5 == 0)
		{
			mat[i] = i;
		}
		else
		{
			mat[i] = 0;
		}
	}

	//initialize(mat,mat_size,INIT_FOR_SPARSE_METRICS);
	return mat;
}

//simple initialization
void initialize(int * input, const int array_size,
	INIT_PARAM PARAM, int x)
{
	if (PARAM == INIT_ONE)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = 1;
		}
	}
	else if (PARAM == INIT_ONE_TO_TEN)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = i % 10;
		}
	}
	else if (PARAM == INIT_RANDOM)
	{
		time_t t;
		srand((unsigned)time(&t));
		for (int i = 0; i < array_size; i++)
		{
			input[i] = (int)(rand() & 0xFF);
		}
	}
	else if (PARAM == INIT_FOR_SPARSE_METRICS)
	{
		srand(time(NULL));
		int value;
		for (int i = 0; i < array_size; i++)
		{
			value = rand() % 25;
			if (value < 5)
			{
				input[i] = value;
			}
			else
			{
				input[i] = 0;
			}
		}
	}
	else if (PARAM == INIT_0_TO_X)
	{
		srand(time(NULL));
		int value;
		for (int i = 0; i < array_size; i++)
		{
			input[i] = (int)(rand() & 0xFF);
		}
	}
}

void initialize(float * input, const int array_size,
	INIT_PARAM PARAM)
{
	if (PARAM == INIT_ONE)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = 1;
		}
	}
	else if (PARAM == INIT_ONE_TO_TEN)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = i % 10;
		}
	}
	else if (PARAM == INIT_RANDOM)
	{
		srand(time(NULL));
		for (int i = 0; i < array_size; i++)
		{
			input[i] = rand() % 10;
		}
	}
	else if (PARAM == INIT_FOR_SPARSE_METRICS)
	{
		srand(time(NULL));
		int value;
		for (int i = 0; i < array_size; i++)
		{
			value = rand() % 25;
			if (value < 5)
			{
				input[i] = value;
			}
			else
			{
				input[i] = 0;
			}
		}
	}
}

//cpu reduction
int reduction_cpu(int * input, const int size)
{
	int sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += input[i];
	}
	return sum;
}

//cpu transpose
void mat_transpose_cpu(int * mat, int * transpose, int nx, int ny)
{
	for (int  iy = 0; iy < ny; iy++)
	{
		for (int  ix = 0; ix < nx; ix++)
		{
			transpose[ix * ny + iy] = mat[iy * nx + ix];
		}
	}
}

//compare results
void compare_results(int gpu_result, int cpu_result)
{
	printf("GPU result : %d , CPU result : %d \n",
		gpu_result, cpu_result);

	if (gpu_result == cpu_result)
	{
		printf("GPU and CPU results are same \n");
		return;
	}

	printf("GPU and CPU results are different \n");
}


//compare arrays
void compare_arrays(int * a, int * b, int size)
{
	for (int  i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			printf("Arrays are different \n");
			printf("%d - %d | %d \n", i, a[i], b[i]);
			//return;
		}
	}
	printf("Arrays are same \n");
}

void compare_arrays(float * a, float * b, float size)
{
	for (int i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			printf("Arrays are different \n");
			
			return;
		}
	}
	printf("Arrays are same \n");
	
}

void print_time_using_host_clock(clock_t start, clock_t end)
{
	printf("GPU kernel execution time : %4.6f \n",
		(double)((double)(end - start) / CLOCKS_PER_SEC));
}

void printData(char *msg, int *in, const int size)
{
	printf("%s: ", msg);

	for (int i = 0; i < size; i++)
	{
		printf("%5d", in[i]);
		fflush(stdout);
	}

	printf("\n");
	return;
}

void sum_array_cpu(float* a, float* b, float *c, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}


}  //namespace opendip