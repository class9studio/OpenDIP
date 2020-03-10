#ifndef ___OPENDIP_ALGORITHM_H_
#define ___OPENDIP_ALGORITHM_H_
#include "image.h"
#include "common.h"

namespace opendip {
    /* 数字图像基础 */
    // 显示图像
    void ImgShow(Image &src, string title);

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
    GrayImgMap GrayImgCvtMap(Image &src);
    GrayImgMapConst GrayImgCvtMapConst(Image &src);
    vector<ColorImgMap> ColorImgCvtMap(Image &src);
    vector<ColorImgMapConst> ColorImgCvtMapConst(Image &src);

    // 通过旋转角度和旋转中心，返回图像旋转矩阵2x3
    Matrix<double, 2, 3> GetRotationMatrix2D(Point2f center, double angle, double scale);

    // 仿射变换
    Image WarpAffine(Image &src, Matrix<double, 2, 3> transform);

    //欧式距离
    double Distance(double x, double y, double c_x, double c_y);
    
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

    //gamma校正
    Image GammaCorrection(Image &src, double fGamma);

    /* 空间滤波 */
    //n*n矩阵逆时针旋转180
    MatrixXd MatRotate180(MatrixXd m);
    // 图像的卷积
    Image Filter2D(Image &src, MatrixXd &kernel);
    Image Filter2D_Gray(Image &src, MatrixXd &kernel,int padding = 0);
    // 矩阵卷积操作
    MatrixXd FilterMatrix2d(MatrixXd &src, MatrixXd &kernel);

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

    //双边滤波
    Image BilateralFilter(Image &src, int ksize, double sigma_distance, double sigma_pixel);

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

    //canny算法
    Image Canny(Image &src, int sobel_size, double threshold1, double threshold2);

    /* 图像形态学 */
    //图像连通域-二值图像
    int ConnectedComponents(Image &image, Image &labels);

    // 获取结构元素形状
    MatrixXd GetStructuringElement(int shape, int ksize);

    // 腐蚀
    Image MorphErode(Image &src, MatrixXd kernel, int padding = 0);

    // 膨胀
    Image MorphDilate(Image &src, MatrixXd kernel, int padding = 0);

    //开运算
    Image MorphOpen(Image &src, MatrixXd kernel);

    //关运算
    Image MorphClose(Image &src, MatrixXd kernel);

    //形态学梯度
    Image MorphGradient(Image &src, MatrixXd kernel, Morph_Gradient_Type type);

    //顶帽运算
    Image MorphTophat(Image &src, MatrixXd kernel);

    //黑帽运算
    Image MorphBlackhat(Image &src, MatrixXd kernel);

    //击中击不中运算
    Image MorphHitMiss(Image &src, MatrixXd kernel);

    /* 空间滤波 */
    //单变量离散傅里叶变换、反变换
    void DFT1D(double* src, Complex* dst, int size);
    void IDFT1D(Complex* src, Complex* dst, int size);

    //二维离散傅里叶变换、反变换
    void DFT2D(double** src, Complex** dst, int w, int h);
    void IDFT2D(Complex** src, Complex** dst, int w, int h);

    //单变量快速傅里叶变换
    void FFT1D(Complex* src, Complex* dst, int size);
    void IFFT1D(Complex* src, Complex* dst, int size);  

    //二维快速傅里叶变换
    void FFT2D(double *src, Complex *dst, int size_w, int size_h);
    void IFFT2D(Complex *src, Complex *dst, int size_n);

    //图像快速傅里叶变换、反变换
    void FFT_Shift(double *src, int size_w, int size_h);
    void ImgFFT(Image &src, Complex *dst);
    Image ImgIFFT(Complex *src, int size_w, int);
    //幅度谱归一化
    void Nomalsize(double *src,double *dst,int size_w,int size_h);
    //通过复数获取频谱图像
    Image GetAmplitudespectrum(Complex * src,int size_w,int size_h);
    
    //滤波器
    void IdealLPFilter(double *Filter, int width, int height, double cut_off_frequency);
    //频域滤波函数
    Image FrequencyFiltering(Image &src, Frequency_Filter_Type filter_type, double param1,int param2);

    /* 模式识别 */
    /* 特征提取算子 */
    // harris角点检测-opencv
    int HarrisCornelDetector(string filename);
    // harris角点检测-opendip
    Image DetectHarrisCorners(Image &src, double alpha, bool with_nms, double threshold);

    // SIFT,SUFR特征点匹配-opencv
    void SurfPicsMatch(string pic1, string pic2);

    // Hog特征提取-opencv
    int HogFeatures(string pic_name);
    // 为每个cell构建直方图
    vector<double> CellHistogram(MatrixXd cell_m, MatrixXd cell_d, int bin_size);
    // Hog特征提取-opendip
    vector<vector<vector<double>>> DetectHOGDescription(Image &src, int cell_size, int bin_size);
    // Hog+SVM用于行人检测
    int HogSvm_PeopleDetector(string pic_name);
    // 原始LBP特征提取
    Image DetectOriginLBP(Image &src);
    // 改进LBP-圆形LBP可以设置半径和采样点
    Image DetectCircleLBP(Image &src, int radius, int neighbors);
} 

#endif //___OPENDIP_ALGORITHM_H_