#ifndef ___OPENDIP_ALGORITHM_H_
#define ___OPENDIP_ALGORITHM_H_
#include "image.h"
#include "common.h"

namespace opendip {
    /* 数字图像基础 */
    // 显示图像
    void ImgShow(Image &src);
    
    // 最邻近插值法(Nearest Interpolation)
    Image LinearInterpolation(Image &src_image, int resize_row, int resize_col);

    // 双线性插值法(Bilinear Interpolation)
    Image BilinearInterpolation(Image &src_image, int resize_w, int resize_h);

    // sperate one channel from image
    Image Split(Image &src, OpenDIP_Channel_Type channel);

    // sperate channels from image
    vector<Image> Split(Image &src);

    // merge channels to image
    Image Merge(vector<Image> &channels, int num);

    // mean and stddev in one channel image
    void MeanStddev(Image &src, double *mean, double *stddev);

    // max and min gray in one channel image
    void MinMaxLoc(Image &src, unsigned char *min, unsigned char *max, Point &min_loc, Point &max_loc);

    // 单通道图像数据映射到Map中
    MapType ImageCvtMap(Image &src);
    MapTypeConst ImageCvtMapConst(Image &src); 

    // 通过旋转角度和旋转中心，返回图像旋转矩阵2x3
    Matrix<double, 2, 3> GetRotationMatrix2D(Point2f center, double angle, double scale);

    // 仿射变换
    Image WarpAffine(Image &src, Matrix<double, 2, 3> transform);

    /* 灰度变换 */
    // color to grayscale conversion
    Image ColorCvtGray(Image &src, OpenDIP_ColorCvtGray_Type cvt_type);

    // Ostu计算阈值
    unsigned char GetOstu(Image &src);
    // Image Binarization
    Image Threshold(Image &src, Thresh_Binary_Type type, double threshold, double max_value, bool auto_threshold);

    //图像的直方图均衡
    Image HistEqualizationGray(Image &src);  //灰度图像
    Image HistEqualization(Image &src);      //灰度或者彩色图像

	//灰度图像的直方图配准
	Image HistRegistration(Image &src);

    //椒盐噪声函数
    void SaltAndPepper(Image &src, int n);

    /* 空间滤波 */
    //n*n矩阵逆时针旋转180
    MatrixXd MatRotate180(MatrixXd m);
    // 图像的卷积
    Image Filter2D_Gray(Image &src, Matrix3d &kernel);  
    Image Filter2D_3M(Image &src, Matrix3d &kernel);  
    Image Filter2D(Image &src, MatrixXd &kernel);
    
    //高斯分布随机数生成
    double RandomGuassinGen(double mean, double sigma);
    //高斯噪声函数
    void GussianNoiseImg_Gray(Image &src, double mean, double sigma);
    void GussianNoiseImg(Image &src, double mean, double sigma);

    //均值滤波
    Image Blur(Image &src, int ksize);

    //二阶高斯滤波器
    MatrixXd GetGaussianKernel(int  ksize, double  sigma);
    //高斯滤波
    Image GaussianBlur(Image &src, int ksize, double sigma);

    /* 图像分割 */
    //边缘检测
    Image EdgeDetection(Image &src, MatrixXd &kernel);                     //单方向边缘检测滤波器
    Image EdgeDetection(Image &src, MatrixXd &kernelX, MatrixXd &kernelY); //两个方向边缘结果相加得到整幅图像的边缘信息

    //Sobel算子构造
    void GetSobel(int n, MatrixXd &sobX, MatrixXd &sobY);
    //Sobel算子 边缘检测
	Image Sobel(Image &src, int ksize = 3);

    //Scharr算子
    Image Scharr(Image &src);

    //Laplacian算子
	Image Laplacian(Image &src);

    /* 图像形态学 */
    //图像连通域-二值图像
    int ConnectedComponents(Image &image, Image &labels);

} 

#endif //___OPENDIP_ALGORITHM_H_