#include <iostream>
#include <Eigen/Dense>
//#include <opencv2/opencv.hpp>
#include "common.h"
#include "image.h"

#define CATCH_CONFIG_MAIN          //catch2的main函数
#include "catch2.h"

//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"
//
//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"

using namespace Eigen;
using namespace opendip;
//using namespace cv;

TEST_CASE( "simple" )
{
    REQUIRE( OPENDIP_IMAGE_PNG == GetImageTypeFromFile((char *)"yanglin.png") );
}

TEST_CASE("eigen")
{       
	 MatrixXd m(2,2);
	 m(0,0) = 3;
	 m(1,0) = 2.5;
	 m(0,1) = -1;
	 m(1,1) = m(1,0) + m(0,1);
	 std::cout << "Here is the matrix m:\n" << m << std::endl;
	 VectorXd v(2);
	 v(0) = 4;
	 v(1) = v(0) - 1;
	 std::cout << "Here is the vector v:\n" << v << std::endl;
	
	REQUIRE( 0 == 0 );
}


#if 0
TEST_CASE("opencv")
{       
    Mat picture = imread("../data/test_image/cat.jpg");
    imshow("OpenCV Test", picture); 
    waitKey(5000);
    REQUIRE( 0 == 0 );
} 
#endif

//TEST_CASE("stb")
//{       
//    int w, h, n;
//
//#if _WIN32
//	char img_path[] = "../../data/test_image/cat.jpg";
//#else
//	char img_path[] = "../data/test_image/aloeGT11.png";
//#endif
//
//    //rgba
//    //load image
//    unsigned char *data = stbi_load(img_path, &w, &h, &n, 0);
//
//
//    std::cout << "aloeGT Info :" << std::endl << "width:" << w << " hight: " << h << " channel: " << n << std::endl;
//
//	//std::cout << "data:" << &data << std::endl;
//	printf("data[0]:%x\n", &data[0]);
//	printf("data[w-1]:%x\n", &data[w-1]);
//	printf("data[w]:%x\n", &data[w]);
//	printf("width:%d", &data[w] - &data[0]);
//
//	//for (size_t j = 0; j < h; j++)
//	//{
//	//	for (size_t i = 0; i < w; i++)
//	//	{
//	//		data[j*n*w + n * i + 0] = 0;
//	//		data[j*n*w + n * i + 1] = 0;
//	//		data[j*n*w + n * i + 2] = 255;
//	//	}
//	//}
//
//	stbi_write_jpg("result..jpg", w, h, n, data, w * n);
//
//    stbi_image_free(data);
//
//	system("pause");
//
//    REQUIRE( 0 == 0 );
//}
TEST_CASE("image read ")
{
	int w, h, n;

#if _WIN32
	char img_path[] = "../../data/test_image/cat.jpg";
#else
	char img_path[] = "../data/test_image/aloeGT11.png";
#endif

	//rgba
	//load image
	//unsigned char *data = stbi_load(img_path, &w, &h, &n, 0);

	Image src = ImgRead(img_path);

	unsigned char *data = (unsigned char *)src.data;

	printf("data[0]:%x\n", &data[0]);
	printf("%d	%d	%d\n", data[100], data[200], data[300]);

	std::cout << "aloeGT Info :" << std::endl << "width:" << src.w << " hight: " << src.h << " channel: " << src.c << std::endl;

	////std::cout << "data:" << &data << std::endl;
	//printf("data[0]:%x\n", &data[0]);
	//printf("data[w-1]:%x\n", &data[w - 1]);
	//printf("data[w]:%x\n", &data[w]);
	//printf("width:%d", &data[w] - &data[0]);
	//for (size_t j = 0; j < h; j++)
	//{
	//	for (size_t i = 0; i < w; i++)
	//	{
	//		data[j*n*w + n * i + 0] = 0;
	//		data[j*n*w + n * i + 1] = 0;
	//		data[j*n*w + n * i + 2] = 255;
	//	}
	//}

	//stbi_write_jpg("result.jpg", src.w, src.h, src.c, data, src.w * src.c);

	//stbi_image_free(data);
	ImgWrite((char*)"result.jpg", src);

	system("pause");

	REQUIRE(0 == 0);
}



