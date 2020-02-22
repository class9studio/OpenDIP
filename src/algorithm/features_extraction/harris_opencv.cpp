#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

#include <string>

#include "common.h"
#include "algorithm.h"
#include "matplotlibcpp.h"

using namespace cv;
using namespace std;

namespace opendip {
int thresh = 80;
Mat src, dst,norm_dst,gray_img,abs_dst,out1,out2;
void callback(int, void*);
int harris_cornel_detector(string filename)
{	
	src = imread(filename);
	namedWindow("input",CV_WINDOW_AUTOSIZE);
	imshow("input", src);
    cvtColor(src, gray_img, CV_BGR2GRAY);
 
	namedWindow("output", CV_WINDOW_AUTOSIZE);
	createTrackbar("threshold", "output", &thresh, 255, callback);
	callback(0, 0);
	waitKey(0);
	return 0;
}
void callback(int, void*) 
{
	dst = Mat::zeros(gray_img.size(), CV_32FC1);
	out1 = Mat::zeros(gray_img.size(), CV_32FC1);
 
	cornerHarris(gray_img, dst, 2, 3, 0.04);
	normalize(dst, norm_dst, 0, 255, NORM_MINMAX, CV_32FC1,Mat());
	convertScaleAbs(norm_dst, abs_dst);
 
	Mat result_img = src.clone();
	for (int i = 0; i < result_img.rows; i++) 
    {
		for (int j = 0; j < result_img.cols; j++) 
        {
			uchar value = abs_dst.at<uchar>(i, j);
			if (value > thresh) 
            {
				circle(result_img, cv::Point(j, i),1,Scalar(0,255,0),2);
			}
		}
	}
	imshow("output", result_img);
}

} //opendip namespace