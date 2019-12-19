#include <iostream>
#include <string>

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

TEST_CASE("image read")
{
	int w, h, n;
	std::string img_path = "../data/test_image/cat.jpg";
	std::string dst_img = "../data/test_image/yl.jpg";
	Image src = ImgRead((char*)img_path.c_str());
	ImgWrite((char*)dst_img.c_str(), src);
 
	REQUIRE(true);
}



