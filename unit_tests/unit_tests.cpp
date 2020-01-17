#include <iostream>
#include <string>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
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
using namespace std;
using namespace opendip;
using namespace cv;

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

#endif
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

TEST_CASE("OpenDIP")
{
	Image src = ImgRead("../data/test_image/lena.jpg");
	vector<Image> dst = Split(src);
	unsigned char val = GetOstu(dst[0]);
	printf("value: %d\n", val);

	Image dst_img = Threshold(dst[0], THRESH_BINARY, 125, 255, false);
	ImgWrite("../data/output_image/linux/lena_ostu.jpg", dst_img);
	REQUIRE(true);
}


TEST_CASE("opendip-验证map函数")
{
	unsigned char ary1[4] = {1,2,3,4};
	unsigned char ary2[4] = {10,20,30,40};
	cout << "opendip-map" << endl;
	MapType src_m1 = MapType((unsigned char *)ary1, 2, 2);
	MapType src_m2 = MapType((unsigned char *)ary2, 2, 2);
	src_m1 = src_m1 + src_m2;

	for(int i = 0; i < 4; i++)
	{
		//cout << ary1[i] << endl;
		printf("value: %d.\n", ary1[i]);
	}

	REQUIRE(true);
}

//测试仿射变换
TEST_CASE("OpenDIP")
{
    Mat img = imread("../data/test_image/lena.jpg");

    Mat rotation0, rotation1, img_warp0, img_warp1;
    double angle = 30; //设置图像旋转的角度
    Size dst_size(img.rows, img.cols); //设置输出图像的尺寸

    cv::Point2f center(img.rows / 2.0, img.cols / 2.0); //设置图像的旋转中心
    rotation0 = getRotationMatrix2D(center, angle, 1); //计算放射变换矩阵

	cout << "rotation0=\n" << rotation0 << endl;
    warpAffine(img, img_warp0, rotation0, dst_size); //进行仿射变换
    imshow("img_warp0", img_warp0);
    //根据定义的三个点进行仿射变换
    cv::Point2f src_points[3];
    cv::Point2f dst_points[3];
    src_points[0] = cv::Point2f(0, 0); //原始图像中的三个点
    src_points[1] = cv::Point2f(0, (float)(img.cols - 1));
    src_points[2] = cv::Point2f((float)(img.rows - 1), (float)(img.cols - 1));

    waitKey(0);
    
	REQUIRE(true);
}

TEST_CASE("仿射变换")
{
	Image src = ImgRead("../data/test_image/lena.jpg");

	opendip::Point2f center(src.h / 2, src.w / 2);
	Matrix<double, 2, 3, RowMajor> rotation =  GetRotationMatrix2D(center, 30,  1);
	cout << "rotation=\n" << rotation << endl;

	Image dst = WarpAffine(src, rotation);
	ImgWrite("../data/output_image/linux/lena_affain.jpg", dst);

	REQUIRE(true);
}

TEST_CASE("opencv-图像卷积")
{
	//待卷积矩阵
	uchar points[25] = { 1,2,3,4,5,
		6,7,8,9,10,
		11,12,13,14,15,
		16,17,18,19,20,
		21,22,23,24,25 };
	Mat img(5, 5, CV_8UC1, points);
	//卷积模板
	Mat kernel = (Mat_<float>(3, 3) << 1, 2, 1,
		2, 0, 2,
		1, 2, 1);
	Mat kernel_norm = kernel / 12; //卷积模板归一化
					//未归一化卷积结果和归一化卷积结果
	Mat result, result_norm;
	filter2D(img, result, CV_32F, kernel, cv::Point(-1, -1), 2, BORDER_CONSTANT);
	filter2D(img, result_norm, CV_32F, kernel_norm, cv::Point(-1,-1),2, BORDER_CONSTANT);
	cout << "result:" << endl << result << endl;
	cout << "result_norm:" << endl << result_norm << endl;	

    //图像卷积
    Mat lena = imread("../data/test_image/lena.jpg");
    if (lena.empty())
    {
      cout << "请确认图像文件名称是否正确" << endl;
    }
    Mat lena_fillter;
    filter2D(lena, lena_fillter, -1, kernel_norm, cv::Point(-1, -1), 2, BORDER_CONSTANT);
    imshow("lena_fillter", lena_fillter);
    imshow("lena", lena);
    waitKey(0);	

	REQUIRE(true);
}

TEST_CASE("opendip-图像卷积")
{
	Image src = ImgRead("../data/test_image/lena.jpg");
	Matrix3d m;
	m <<  1, 2, 1,
		  2, 0, 2,
		  1, 2, 1;
	m = m/12;

	Image dst = Filter2D(src, m);
	ImgWrite("../data/output_image/linux/lena_conva_color.jpg", dst);

	REQUIRE(true);
}


TEST_CASE("opendip-椒盐噪声")
{
	Image src = ImgRead("../data/test_image/lena.jpg");
	SaltAndPepper(src, 10000);

	ImgWrite("../data/output_image/linux/lena_noise.jpg", src);
	REQUIRE(true);
}

TEST_CASE("opendio-高斯噪声")
{
	//灰度图像高斯
	Image src = ImgRead("../data/test_image/aloeGT.png");	
	GussianNoiseImg_Gray(src, 15, 30);
	ImgWrite("../data/output_image/linux/aloeGT_gussian_noise.jpg", src);

	//彩色图像高斯
	Image src1 = ImgRead("../data/test_image/lena.jpg");
	GussianNoiseImg(src1, 0, 5);
	ImgWrite("../data/output_image/linux/lena_gussian_noise.jpg", src1);

	REQUIRE(true);
}

TEST_CASE("opendip-Rotate Matrix")
{
	Matrix3d m;
	m <<  1, 2, 3,
		  4, 0, 2,
		  7, 8, 9;
	
	MatrixXd m1 = MatRotate180(m);
	cout << m1 << endl;

	REQUIRE(true);
}


TEST_CASE("opendip-Rotate Matrix")
{
	MatrixXd m(3,3);
	m <<  1, 2, 3,
		  4, 0, 2,
		  7, 8, 9;
	
	MatrixXd m1(3,3);
	m1 << 1,2,3,
		  4,5,6,
		  7,8,9;

	MatrixXd res = m.array() * m1.array();	
	cout << res << endl;
	cout << "sum: " << res.sum() << endl;
	REQUIRE(true);
}

TEST_CASE("opendip-均值滤波")
{
	Image src = ImgRead("../data/test_image/lena.jpg");

	Image dst = Blur(src, 3);
	ImgWrite("../data/output_image/linux/lena_blur_3.jpg", dst);

	Image dst1 = Blur(src, 5);
	ImgWrite("../data/output_image/linux/lena_blur_5.jpg", dst1);

	Image dst2 = Blur(src, 9);
	ImgWrite("../data/output_image/linux/lena_blur_9.jpg", dst2);
	REQUIRE(true);
}

TEST_CASE("opendip-高斯滤波器")
{
	MatrixXd m = GetGaussianKernel(3, 1);
	cout << "kernel gaussian: " << endl;
	//cout << m << endl;

	Image src = ImgRead("../data/test_image/lena.jpg");
	Image dst = GaussianBlur(src, 3, 1);
	ImgWrite("../data/output_image/linux/lena_gau_blur.jpg", dst);

	Image dst9 = GaussianBlur(src, 9, 1);
	ImgWrite("../data/output_image/linux/lena_gau_blur9.jpg", dst9);
	REQUIRE(true);
}


TEST_CASE("opendip-边缘检测")
{
	Image src = ImgRead("../data/test_image/lena_gray.jpg");
	MatrixXd mX(1,3);
	mX << 1, 0, -1;
	MatrixXd mY(3,1);
	mY << 1, 0, -1;

	Image dst = EdgeDetection(src, mX, mY); 
	ImgWrite("../data/output_image/linux/lena_edg_whole.jpg", dst);

	REQUIRE(true);
}
#endif

TEST_CASE("opendip-sobel")
{
	MatrixXd sobX(5,5), sobY(5,5);
	GetSobel(5, sobX, sobY);
	cout << "sobelX: " << endl;
	cout << sobX << endl;
	cout << "sobelY: " << endl;
	cout << sobY << endl;
	
	REQUIRE(true);
}

TEST_CASE("opendip-图像卷积")
{
	Image src = ImgRead("../data/test_image/lena_gray.jpg");
	Image dst = EdgSobel(src, 3);

	ImgWrite("../data/output_image/linux/lena_sobel.jpg", dst);
	REQUIRE(true);
}

