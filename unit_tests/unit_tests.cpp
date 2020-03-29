#include <iostream>
#include <string>

#include "common.h"
#include "image.h"
#include "algorithm.h"
#include "timing.h"
#include "cudahead.h" 

#define CATCH_CONFIG_MAIN          //catch2的main函数
#include "catch2.h"

#include <Eigen/Dense>

#if  defined(__linux)
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "matplotlibcpp.h"
using namespace cv;
namespace plt = matplotlibcpp;    //图库matplotlib-cpp头文件
#endif

using namespace Eigen;
using namespace std;
using namespace opendip;

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

TEST_CASE("opendip-边缘检测")
{
	Image src = ImgRead("../data/test_image/lena_gray.jpg");
	MatrixXd mX(1,3);
	mX << 1, 0, -1;
	MatrixXd mY(3,1);
	mY << 1, 0, -1;

	Image dst = EdgeDetection(src, mX, mY); 
	ImgWrite("../data/output_image/linux/lena_edg.jpg", dst);

	REQUIRE(true);
}

TEST_CASE("opendip-图像卷积")
{
	Image src = ImgRead("../data/test_image/lena_gray.jpg");
	Image dst = Sobel(src, 3);

	ImgWrite("../data/output_image/linux/lena_sobel.jpg", dst);
	REQUIRE(true);
}


TEST_CASE("opendip-Schaar")
{
	Image src = ImgRead("../data/test_image/lena_gray.jpg");
	Image dst = Scharr(src);

	ImgWrite("../data/output_image/linux/lena_scharr.jpg", dst);
	REQUIRE(true);
}

TEST_CASE("opendip-lapcian")
{
	Image src = ImgRead("../data/test_image/lena_gray.jpg");
	Image dst = GaussianBlur(src, 3, 1);

	Image dst_lap = Laplacian(dst);
	ImgWrite("../data/output_image/linux/lena_lanp0.jpg", dst_lap);
	REQUIRE(true);
}

TEST_CASE("OpenDIP-连通域")
{
	Image src = ImgRead("../data/test_image/rice.png");
	Image dst_img = Threshold(src,  opendip::THRESH_BINARY, 125, 255, false);
	ImgWrite("../data/output_image/linux/rice_binary.jpg", dst_img);

	Image labels(dst_img.w, dst_img.h, dst_img.c);
	int num = ConnectedComponents(dst_img, labels);
	cout << "connect components numbers: " << num << endl;
	ImgWrite("../data/output_image/linux/rice_lianton.jpg", labels);

	REQUIRE(true);
}

TEST_CASE("opendip-matplot测试")
{
	Image src = ImgRead("../data/test_image/lena.jpg");
	ImgShow(src);

	REQUIRE(true);
}

TEST_CASE("opendip-图像卷积")
{
	Image src = ImgRead("../data/test_image/lena.jpg");
	MatrixXd m(3,3);
	m <<  1, 2, 1,
		  2, 0, 2,
		  1, 2, 1;
	m = m/12;

	Image dst = Filter2D(src, m);
	ImgShow(dst, "lena");
	ImgWrite("../data/output_image/linux/lena_conva_color1.jpg", dst);

	REQUIRE(true);
}


TEST_CASE("opendip-腐蚀-膨胀")
{
	#if 0
	Image src(6,6,1);
	
	MapType img_m = ImageCvtMap(src);
	img_m <<    0, 0, 0, 0, 255, 0,
				0, 255, 255, 255, 255, 255,
				0, 255, 255, 255, 255, 0,
				0, 255, 255, 255, 255, 0,
				0, 255, 255, 255, 255, 0,
				0, 0, 0, 0, 0, 0;
	#endif

	Image src = ImgRead("../data/test_image/Morphology_Original.png");
	MatrixXd m = GetStructuringElement(1, 3);

	ImgShow(src, "source");
	double startTime = now();
	Image dst = MorphDilate(src, m);
	double nDetectTime = calcElapsed(startTime, now());
    printf("Dilate time: %d ms.\n ", (int)(nDetectTime * 1000));
	ImgShow(dst, "My Dilate");

	startTime = now();
	Image dst_erode = MorphErode(src, m);
	nDetectTime = calcElapsed(startTime, now());
    printf("Erode time: %d ms.\n ", (int)(nDetectTime * 1000));
	ImgShow(dst_erode, "My Erode");
	REQUIRE(true);
}

TEST_CASE("[test-2] opendip-图像卷积")
{
	Image src = ImgRead("../data/test_image/lena_gray.jpg");		

	MatrixXd kernel(3,3);
	kernel<< 1, 2, 1,
			 2, 0, 2,
			 1, 2, 1;
	kernel = kernel / 12;
	ImgShow(src, "lena");
	double startTime = now();
	Image dst = Filter2D(src, kernel);
	double nDetectTime = calcElapsed(startTime, now());
    printf("conv1 time: %d ms.\n ", (int)(nDetectTime * 1000));
	ImgShow(dst, "My Cov1");

	startTime = now();
	Image dst1 = Filter2D_Gray(src, kernel);
	nDetectTime = calcElapsed(startTime, now());
    printf("conv2 time: %d ms.\n ", (int)(nDetectTime * 1000));
	ImgShow(dst1, "My Cov2");

	REQUIRE(true);
}

TEST_CASE("opendip-图形开、关运算")
{
	Image src(12,9,1);
	
	MapType src_m = ImageCvtMap(src);
	src_m <<    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0,
				0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
				0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
				0, 255, 255, 255, 0, 255, 255, 255, 0, 0, 0, 0,
				0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
				0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0,
				0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	ImgShow(src, "source img");

	MatrixXd m = GetStructuringElement(0, 3);	
	Image dst_open = MorphOpen(src, m);
	ImgShow(dst_open, "dst open");

	Image dst_close = MorphClose(src, m);
	ImgShow(dst_close, "dst close");

	Image dst_gradient = MorphGradient(src, m, MORPH_GRADIENT_BASIC);
	ImgShow(dst_gradient, "dst gradient");

	Image dst_top_hat = MorphTophat(src, m);
	ImgShow(dst_top_hat, "dst top hat");

	Image dst_bla_hat = MorphBlackhat(src, m);
	ImgShow(dst_bla_hat, "dst black hat");

	Image dst_hit_miss = MorphHitMiss(src, m);
	ImgShow(dst_hit_miss, "dst hit miss");
	
	REQUIRE(true);
}

TEST_CASE("opendip-图形开、关运算")
{
	Image src = ImgRead("../data/test_image/lena.jpg");
	MatrixXd m = GetStructuringElement(0, 3);
	Image dst_gradient = MorphGradient(src, m, MORPH_GRADIENT_BASIC);
	ImgShow(dst_gradient, "dst gradient");
	REQUIRE(true);
}

TEST_CASE("[test-1] opendip-Raw Map")
{
	Image src(2,2,3);
	int count = 0;
	unsigned char *p_src = (unsigned char*)src.data;
	for(int j = 0; j < 2; j++)
	{
		for(int i = 0; i < 2; i++)
		{
			for(int z = 0; z < 3; z++)
			{
				p_src[j*src.w*src.c + i*src.c + z] = count;
				cout << count << endl;
				count++;
			}
		}
	}
	vector<ColorImgMap> maps = ColorImgCvtMap(src);
	int size = maps.size();
	int pixel_val = 0;
	for(int j = 0; j < 2; j++)
	{
		for(int i = 0; i < 2; i++)
		{
			pixel_val = maps[0](j,i);
			cout << "channel 0: " << pixel_val << endl;

			pixel_val = maps[1](j,i);
			cout << "channel 1: " << pixel_val << endl;

			pixel_val = maps[2](j,i);
			cout << "channel 2: " << pixel_val << endl;
		}
	}

	REQUIRE(true);
}

TEST_CASE("[test-2] opendip-Color Image Map")
{
	Image src = ImgRead("../data/test_image/lena.jpg");
	vector<ColorImgMap> maps = ColorImgCvtMap(src);
	int size = maps.size();
	cout << "maps size: " << size << endl;

	vector<Image> dst = Split(src);
	ImgShow(dst[0], "R");	
	ImgShow(dst[1], "G");	
	ImgShow(dst[2], "B");	

	for(int j = 0; j < src.h; j++)
	{
		for(int i = 0; i < src.w; i++)
		{
			maps[0](j,i) = 0;
		}
	}
	vector<Image> dst1 = Split(src);
	ImgWrite("../data/output_image/linux/lena_R.jpg", dst1[0]);
	ImgShow(dst1[0], "R");	
	ImgShow(dst1[1], "G");	
	ImgShow(dst1[2], "B");
	REQUIRE(true);
}

TEST_CASE("[test-2] opendip-Gray Image Map")
{
	Image src = ImgRead("../data/test_image/lena_gray.jpg");
	vector<GrayImgMap> maps = GrayImgCvtMap(src);
	int size = maps.size();
	cout << "maps size: " << size << endl;
	ImgShow(src, "Before");
	for(int j = 0; j < maps[0].rows(); j++)
	{
		for(int i = 0; i < maps[0].cols(); i++)
		{
			maps[0](j,i) = 255;
		}
	}

	ImgWrite("../data/output_image/linux/lena_R.jpg", src);
	ImgShow(src, "After");	

	REQUIRE(true);
}

TEST_CASE("opendip-Complex")
{
	opendip::Complex c1, c2, c3;
    cin>>c1>>c2;

	c3 = c1 + c2;
    cout<<"c1 + c2 = "<<c3<<endl;

	c1 = c2;
	cout << "c1: " << c1 << endl;
	cout << "c2: " << c2 << endl;
	REQUIRE(true);
}

TEST_CASE("opencv-harris corner detector")
{
	string filename = "../data/test_image/lena.jpg";
	HarrisCornelDetector(filename);
	REQUIRE(true);
}

TEST_CASE("opendip-BilateralFilter")
{
	Image src = ImgRead("../data/test_image/baboon.jpg");
	ImgShow(src, "Before Color");
	Image src_gray = ColorCvtGray(src, OPENDIP_COLORCVTGRAY_AVERAGE);

	ImgShow(src_gray, "Before");
	Image dst = BilateralFilter(src_gray, 17, 2, 50);
	ImgShow(dst, "After");
	REQUIRE(true);
}

TEST_CASE("opendip-Hog Detector")
{
	//int res = HogFeatures("../data/test_image/lena.jpg");
	//cout << "Hog Features Numbers: " << res << endl;

	double startTime = now();
	int val = HogSvm_PeopleDetector("../data/test_image/vtest.avi_2.jpg");
	double nDetectTime = calcElapsed(startTime, now());
	printf("Hog SVM Detector time: %d ms.\n ", (int)(nDetectTime * 1000));
	REQUIRE(true);
}

TEST_CASE("[test-1] opendip-图像卷积")
{
	Image src(5,5,1);
	
	vector<GrayImgMap>img_m = GrayImgCvtMap(src);
	img_m[0] <<    1,2,3,4,5,
				6,7,8,9,10,
				11,12,13,14,15,
				16,17,18,19,20,
				21,22,23,24,25;

	MatrixXd kernel(3,3);
	kernel<< 1, 2, 1,
			 2, 0, 2,
			 1, 2, 1;

	Image dst = Filter2D(src, kernel);
	unsigned char* p_dst = (unsigned char*)dst.data;
	for(int j = 0; j < dst.h; j++)
	{
		for(int i = 0; i < dst.w; i++)
		{
			int tmp = p_dst[j*dst.c*dst.w + i*dst.c];
			cout << tmp << endl;
		}
	}

	REQUIRE(true);
}

TEST_CASE("opendip-FilterMatrix2d")
{       
	MatrixXd kernel(3,3);
	kernel<< 1, 2, 1,
			 2, 0, 2,
			 1, 2, 1;

	MatrixXd ImgMat(5,5);
	ImgMat << 1,2,3,4,5,
			6,7,8,9,10,
			11,12,13,14,15,
			16,17,18,19,20,
			21,22,23,24,25;

    MatrixXd Mat = FilterMatrix2d(ImgMat, kernel);
	cout << Mat << endl;

	REQUIRE(true);
}

TEST_CASE("opendip-Harris角点检测")
{
	Image src = ImgRead("../data/test_image/lena_gray.jpg");
	ImgShow(src, "Before");
	Image dst = DetectHarrisCorners(src, 0.04, false, 0.01);
	cout << "channels: "<< dst.c << endl;
	ImgShow(dst, "Result");
	REQUIRE(true);
}

TEST_CASE("opendip-Gamma Correction")
{
	Image src = ImgRead("../data/test_image/lena.jpg");
	ImgShow(src, "origin");
	Image dst = GammaCorrection(src, 0.4);
	ImgShow(dst, "gamma1");
	Image dst1 = GammaCorrection(src, 2.5);
	ImgShow(dst1, "gamma2");
	REQUIRE(true);
}

TEST_CASE("opendip-OriginLbp")
{
	Image src = ImgRead("../data/test_image/lena_gray.jpg");
	ImgShow(src, "origin");
	Image dst = DetectOriginLBP(src);
	ImgShow(dst, "origin lbp");
}

TEST_CASE("opendip-CircleLbp")
{
	Image src = ImgRead("../data/test_image/lena_gray.jpg");
	ImgShow(src, "origin");
	Image dst = DetectCircleLBP(src, 3, 8);
	ImgShow(dst, "circle lbp");
	Image dst1 = DetectCircleLBP(src, 1, 8);
	ImgShow(dst1, "circle1 lbp");
}

TEST_CASE("cuda")
{
	cudaDeviceTest();
	REQUIRE(true);
}

TEST_CASE("cuda-vecAdd")
{
	int N = 1024;
	cudaVecAddTest(N);
	REQUIRE(true);
}

TEST_CASE("opendip-cudaStencil")
{
	int N = 30;
	cudaStencilTest(N);
	REQUIRE(true);
}
#endif

TEST_CASE("algorithm-cuda resize compare")
{
	Image src = ImgRead("../data/test_image/lena.jpg");

	double startTime = now();
	Image dst_bilinear = BilinearInterpolation(src, 1856, 960);
	double nDetectTime = calcElapsed(startTime, now());
    printf("cpu BilinearInterpolation time: %d ms.\n ", (int)(nDetectTime * 1000));
	ImgShow(dst_bilinear, "cpu_bilinear");


	Image dst(1856, 960, src.c);
	uchar *p_src_data = (uchar *)src.data;
	uchar *p_dst_data = (uchar *)dst.data;
	
	
	startTime = now();
	cudaResize(p_src_data, src.w, src.h, 1856, 960, src.c, &p_dst_data);
	nDetectTime = calcElapsed(startTime, now());
    printf("gpu BilinearInterpolation time: %d ms.\n ", (int)(nDetectTime * 1000));	
	ImgShow(dst, "gpu_bilinear");

	REQUIRE(true);
}











