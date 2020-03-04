#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace opendip
{

int HogFeatures(string pic_name)
{
    Mat src = imread(pic_name);
    if(src.empty())
    {
        cout << "Load image error!" << endl;
        return -1;
    }

    Mat dst, dst_gray;
    resize(src, dst, Size(64, 128));
    cvtColor(dst, dst_gray, COLOR_BGR2GRAY);

    // Hog 特征提取: 检测窗口大小128x64    块：16x16  单元格: 8x8  梯度方向分成9个方向
    HOGDescriptor detector(Size(64, 128), Size(16,16), Size(8,8), Size(8,8), 9);

    vector<float> descriptors;
    vector<Point> locations;
    detector.compute(dst_gray, descriptors, Size(0, 0), Size(0, 0), locations);

    return descriptors.size();
}

int HogSvm_PeopleDetector(string pic_name)
{
	Mat src = imread(pic_name);
	if (src.empty()) 
	{
		cout << "Load image error!" << endl;
		return -1;
	}

    HOGDescriptor hog = HOGDescriptor();
    hog.setSVMDetector(hog.getDefaultPeopleDetector());

    vector<Rect> foundLocations;
    hog.detectMultiScale(src, foundLocations, 0, Size(8, 8), Size(32, 32), 1.05, 2);
    Mat result = src.clone();
    // 画目标框
	for (size_t t = 0; t < foundLocations.size(); t++) 
		rectangle(result, foundLocations[t], Scalar(0, 0, 255), 2, 8, 0);

    imshow("HOG SVM Detector Demo", result);
    waitKey();
    return 0;
}

}