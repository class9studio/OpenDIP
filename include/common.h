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
#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>
#include <utime.h>
#include <fstream> 

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
    typedef Map<RowMatrixXc, Unaligned, InnerStride<1> > GrayImgMap;
    typedef Map<const RowMatrixXc, Unaligned, InnerStride<1> > GrayImgMapConst;

    typedef Map<RowMatrixXc, Unaligned, InnerStride<3> > ColorImgMap;
    typedef Map<const RowMatrixXc, Unaligned, InnerStride<3> > ColorImgMapConst;

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

    //复数结构体
    struct Complex 
    {
        double r, i;
        Complex() { r = 0, i = 0; }
        Complex(double real, double imag): r(real), i(imag) {}
    };

    Complex operator + (Complex a, Complex b);
    Complex operator - (Complex a, Complex b);
    Complex operator * (Complex a, Complex b);
    istream& operator >> (istream &in, Complex &a);
    ostream& operator << (ostream &out, Complex &a);

    enum Frequency_Filter_Type
    {
        FRE_FILTER_ILPF = 0x0,   //理想低通滤波器
        FRE_FILTER_BLPF,         //布特沃斯低通滤波器
        FRE_FILTER_GLPF,         //高斯低通滤波器
    };


    #define HANDLE_NULL( a ){if (a == NULL) { \
                                printf( "Host memory failed in %s at line %d\n", \
                                        __FILE__, __LINE__ ); \
                                exit( EXIT_FAILURE );}}

    enum INIT_PARAM{
        INIT_ZERO,INIT_RANDOM,INIT_ONE,INIT_ONE_TO_TEN,INIT_FOR_SPARSE_METRICS,INIT_0_TO_X
    };

    //simple initialization
    void initialize(int * input, const int array_size,
        INIT_PARAM PARAM = INIT_ONE_TO_TEN, int x = 0);

    void initialize(float * input, const int array_size,
        INIT_PARAM PARAM = INIT_ONE_TO_TEN);

    void launch_dummmy_kernel();

    //compare two arrays
    void compare_arrays(int * a, int * b, int size);

    //reduction in cpu
    int reduction_cpu(int * input, const int size);

    //compare results
    void compare_results(int gpu_result, int cpu_result);

    //print array
    void print_array(int * input, const int array_size);

    //print array
    void print_array(float * input, const int array_size);

    //print matrix
    void print_matrix(int * matrix, int nx, int ny);

    void print_matrix(float * matrix, int nx, int ny);

    //get matrix
    int* get_matrix(int rows, int columns);

    //matrix transpose in CPU
    void mat_transpose_cpu(int * mat, int * transpose, int nx, int ny);

    //print_time_using_host_clock
    void print_time_using_host_clock(clock_t start, clock_t end);

    void printData(char *msg, int *in, const int size);

    void compare_arrays(float * a, float * b, float size);

    void sum_array_cpu(float* a, float* b, float *c, int size);

    void print_arrays_toafile(int*, int , char* );

    void print_arrays_toafile_side_by_side(float*,float*,int,char*);

    void print_arrays_toafile_side_by_side(int*, int*, int, char*);

}; // namespace opendip
#endif
