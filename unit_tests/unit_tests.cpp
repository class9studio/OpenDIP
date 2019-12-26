#include <iostream>
#include <string>

#include <Eigen/Dense>
//#include <opencv2/opencv.hpp>
#include "common.h"
#include "image.h"
#include "algorithm.h"
#include "timing.h"

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

#if 0
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



TEST_CASE("opencv")
{       
    Mat picture = imread("../data/test_image/cat.jpg");
    imshow("OpenCV Test", picture); 
    waitKey(5000);
    REQUIRE( 0 == 0 );
} 


TEST_CASE("image read")
{

#if _WIN32
	std::string img_path = "../../data/test_image/cat.jpg";
	std::string dst_img = "../../data/output_image/windows/cat_copy.jpg";
#else
	std::string img_path = "../data/test_image/cat.jpg";
	std::string dst_img = "../data/output_image/linux/cat_copy.jpg";
#endif
	Image src = ImgRead((char*)img_path.c_str());
	Image dst(src);//浅拷贝
	unsigned char* p_data =(unsigned char*) dst.data;
	for (size_t j = 0; j < dst.h; j++)
	{
		for (size_t i = 0; i < dst.w; i++)
		{
			p_data[j * dst.c * dst.w + dst.c*i + 0] = 255;
			p_data[j * dst.c * dst.w + dst.c*i + 1] = 0;
			p_data[j * dst.c * dst.w + dst.c*i + 2] = 0;
		}
	}
	ImgWrite((char*)dst_img.c_str(), src);
 
	REQUIRE(true);
}
#endif

TEST_CASE("algorithm-interpolation")
{
	#if _WIN32
		std::string img_path = "../../data/test_image/cat.jpg";
		std::string dst_img = "../../data/output_image/windows/cat_copy.jpg";
	#else
		std::string img_path = "../data/test_image/cat.jpg";
		std::string dst_img_linar = "../data/output_image/linux/cat_linar.jpg";
		std::string dst_img_Bilinear = "../data/output_image/linux/cat_bilinear.jpg";
	#endif	
	
	Image src = ImgRead((char*)img_path.c_str());
	double startTime = now();
	Image dst_linear = LinearInterpolation(src, 800, 600);
	double nDetectTime = calcElapsed(startTime, now());
    printf("LinearInterpolation time: %d ms.\n ", (int)(nDetectTime * 1000));

	startTime = now();
	Image dst_bilinear = BilinearInterpolation(src, 800, 600);
	nDetectTime = calcElapsed(startTime, now());
    printf("BilinearInterpolation time: %d ms.\n ", (int)(nDetectTime * 1000));

	ImgWrite((char*)dst_img_linar.c_str(), dst_linear);
	ImgWrite((char*)dst_img_Bilinear.c_str(), dst_bilinear);
	 REQUIRE( true);
}

#if 0
TEST_CASE("algorithm-splice")
{
	#if _WIN32
		std::string img_path = "../../data/test_image/cat.jpg";
		std::string dst_img = "../../data/output_image/windows/cat_copy.jpg";
	#else
		std::string img_path = "../data/test_image/lena.jpg";
		std::string dst_img_channel0 = "../data/output_image/linux/cat_R.jpg";
		std::string dst_img_channel1 = "../data/output_image/linux/cat_G.jpg";
		std::string dst_img_channel2 = "../data/output_image/linux/cat_B.jpg";
		std::string dst_img_merge = "../data/output_image/linux/cat_merge.jpg";
		std::string dst_img_cvt = "../data/output_image/linux/cat_cvt.jpg";
	#endif	
	
	Image src = ImgRead((char*)img_path.c_str());
	vector<Image> dst = Split(src);


	ImgWrite((char*)dst_img_channel0.c_str(),dst[0]);
	ImgWrite((char*)dst_img_channel1.c_str(),dst[1]);
	ImgWrite((char*)dst_img_channel2.c_str(),dst[2]);

	Image dst_merge = Merge(dst, 3);
	ImgWrite((char*)dst_img_merge.c_str(),dst_merge);

	Image cvt_dst = ColorCvtGray(src, OPENDIP_COLORCVTGRAY_AVERAGE);
	ImgWrite((char*)dst_img_cvt.c_str(),cvt_dst);

	 REQUIRE( true);
}


TEST_CASE("opencv")
{       
    Mat src = imread("../data/output_image/linux/cat_R.jpg", IMREAD_GRAYSCALE);
    if (src.empty()) {
        printf("could not load image...\n");
		REQUIRE( false );
    }
    namedWindow("input", WINDOW_AUTOSIZE);
    imshow("input", src);

    double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc; // 先定义
    minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, Mat()); // 通过引用对变量写入值
    printf("min: %.2f, max: %.2f \n", minVal, maxVal);
    printf("min loc: (%d, %d) \n", minLoc.x, minLoc.y);
    printf("max loc: (%d, %d)\n", maxLoc.x, maxLoc.y);

    // 彩色图像 三通道的 均值与方差
    src = imread("../data/test_image/lena.jpg");
    Mat means, stddev; // 均值和方差不是一个值。对彩色图像是三行一列的mat
    meanStdDev(src, means, stddev);
    printf("blue channel->> mean: %.2f, stddev: %.2f\n", means.at<double>(0, 0), stddev.at<double>(0, 0));
    printf("green channel->> mean: %.2f, stddev: %.2f\n", means.at<double>(1, 0), stddev.at<double>(1, 0));
    printf("red channel->> mean: %.2f, stddev: %.2f\n", means.at<double>(2, 0), stddev.at<double>(2, 0));

    REQUIRE(true);
} 

TEST_CASE("OpenDIP")
{
	Image src = ImgRead("../data/test_image/lena.jpg");
	vector<Image> dst = Split(src);

	unsigned char min = 0,max = 0;
	opendip::Point min_loc;
	opendip::Point max_loc;
	MinMaxLoc(dst[0], &min, &max, min_loc, max_loc);

	double mean = 0;
	double stddev = 0.0;
	MeanStddev(dst[0], &mean, &stddev);

	printf("mean: %.2f, stddev: %.2f \n", mean, stddev);

	REQUIRE(true);
}


TEST_CASE("eigen")
{       
	MatrixXf M1(3,3);    // Column-major storage
	M1 << 1, 2, 3,
		4, 5, 6,
		7, 8, 9;
	Map<RowVectorXf> v1(M1.data(), M1.size());
	cout << "v1:" << endl << v1 << endl;
	Matrix<float,Dynamic,Dynamic,RowMajor> M2(M1);
	Map<RowVectorXf> v2(M2.data(), M2.size());
	cout << "v2:" << endl << v2 << endl;

	cout << "============================================" << endl;
	Image src = ImgRead("../data/test_image/lena.jpg");
	vector<Image> dst = Split(src);
	
	MapType img_map = ImageCvtMap(dst[0]);
	img_map = img_map*2;  //灰度增加一倍

	ImgWrite("../data/output_image/linux/lena_map.jpg", dst[0]);

	REQUIRE( true );
}
#endif