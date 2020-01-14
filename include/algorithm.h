#ifndef ___OPENDIP_ALGORITHM_H_
#define ___OPENDIP_ALGORITHM_H_
#include "image.h"
#include "common.h"

namespace opendip {
    // sperate one channel from image
    Image Split(Image &src, OpenDIP_Channel_Type channel);

    // sperate channels from image
    vector<Image> Split(Image &src);

    // merge channels to image
    Image Merge(vector<Image> &channels, int num);

    // color to grayscale conversion
    Image ColorCvtGray(Image &src, OpenDIP_ColorCvtGray_Type cvt_type);

    // mean and stddev in one channel image
    void MeanStddev(Image &src, double *mean, double *stddev);

    // max and min gray in one channel image
    void MinMaxLoc(Image &src, unsigned char *min, unsigned char *max, Point &min_loc, Point &max_loc);

    // 单通道图像数据映射到Map中
    MapType ImageCvtMap(Image &src);
    MapTypeConst ImageCvtMapConst(Image &src); 

    // Ostu计算阈值
    unsigned char GetOstu(Image &src);
    // Image Binarization
    Image Threshold(Image &src, Thresh_Binary_Type type, double threshold, double max_value, bool auto_threshold);

    // 通过旋转角度和旋转中心，返回图像旋转矩阵2x3
    Matrix<double, 2, 3> GetRotationMatrix2D(Point2f center, double angle, double scale);

    // 仿射变换
    Image WarpAffine(Image &src, Matrix<double, 2, 3> transform);

    //图像的直方图均衡
    Image HistEqualizationGray(Image &src);  //灰度图像
    Image HistEqualization(Image &src);      //灰度或者彩色图像

	//灰度图像的直方图配准
	Image HistRegistration(Image &src);

    // 最邻近插值法(Nearest Interpolation)
    Image LinearInterpolation(Image &src_image, int resize_row, int resize_col);

    // 双线性插值法(Bilinear Interpolation)
    Image BilinearInterpolation(Image &src_image, int resize_w, int resize_h);

    // 图像的卷积
    Image Filter2D_Gray(Image &src, Matrix3d &kernel);  
    Image Filter2D(Image &src, Matrix3d &kernel);  
} 

#endif //___OPENDIP_ALGORITHM_H_