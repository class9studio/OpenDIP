#include <iostream>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

#include "common.h"
#include "algorithm.h"

using namespace std;
using namespace Eigen;
namespace opendip
{
/*****************************************************************************
*   Function name: DetectHarrisCorners
*   Description  : Harris角点检测
*   Parameters   : src                  检测图像图像类
*                  alpha                响应函数的超参数(nice choice: 0.04<=alpha<=0.06)
*                  with_nms             是否需要最大值抑制
*                  threshold            响应函数值阈值比例参数(nice choice: 0.01)
*   Return Value : Image                原始图像大小，角点像素255
*   Spec         :
*         Harris角点检测有旋转不变性， 但是不具备尺寸不变性
*   History:
*
*       1.  Date         : 2020-3-7  1:50
*           Author       : YangLin
*           Modification : Created function
*****************************************************************************/
Image DetectHarrisCorners(Image &src, double alpha, bool with_nms, double threshold)
{
    assert(src.c == 1); //color img should switch to gray pic before use
    int ksize = 3;
    //目标图像，角点处像素255
    Image dst(src.w, src.h, src.c);

    //获取sobel算子
    MatrixXd sobX(ksize,ksize), sobY(ksize,ksize);
    GetSobel(ksize, sobX, sobY);

    // 计算图像I(x,y)在X和Y两个方向的梯度Ix、Iy
    Image dstX = Filter2D(src, sobX);
    Image dstY = Filter2D(src, sobY);    

    unsigned char *p_srcX_data = (unsigned char *)dstX.data;
    unsigned char *p_srcY_data = (unsigned char *)dstY.data;
    unsigned char *p_dst_data = (unsigned char *)dst.data;
    memset(p_dst_data, 0, src.w*src.h);
    // 计算图像两个方向梯度的乘积
    MatrixXd Ix2(src.h, src.w),Iy2(src.h, src.w),Ixy(src.h, src.w);
    for(int i = 0; i < src.h; i++)
    {
        for(int j = 0; j < src.w; j++)
        {
            double valx =  p_srcX_data[i*src.w + j];
            double valy =  p_srcY_data[i*src.w + j];
            Ix2(i,j) =  pow(valx,2);
            Iy2(i,j) =  pow(valy,2);
            Ixy(i,j) =  valx*valy;
        }
    }

    //使用高斯函数对I2x、I2y和Ixy进行高斯加权（取σ取1)
    MatrixXd guassKernel = GetGaussianKernel(7, 1);
    MatrixXd guassIx2 = FilterMatrix2d(Ix2, guassKernel);
    MatrixXd guassIy2 = FilterMatrix2d(Iy2, guassKernel);
    MatrixXd guassIxy = FilterMatrix2d(Ixy, guassKernel);

    //计算局部特征结果矩阵M的特征值和响应函数R(i,j)=det(M)-k(trace(M))^2
    MatrixXd cornerStrength = MatrixXd::Zero(src.h, src.w);
    for(int i = 0; i < src.h; i++)
    {
        for(int j = 0; j < src.w; j++)
        {
			double det_m = guassIx2(i,j) * guassIy2(i,j) - guassIxy(i,j) * guassIxy(i,j);
			double trace_m = guassIx2(i,j) + guassIy2(i,j);
			cornerStrength(i,j) = det_m - alpha * trace_m *trace_m;            
        }
    }

    // 将计算出响应函数的值R进行非极大值抑制，滤除一些不是角点的点，同时要满足大于设定的阈值
    double maxValue = cornerStrength.maxCoeff(); //最大值
    cout << "Max value: " << maxValue << endl;
    for(int i = 0; i < src.h; i++)
    {
        for(int j = 0; j < src.w; j++)
        {
            if(with_nms) //3x3邻域最大值抑制+阈值判断
            {
                if(cornerStrength(i,j)>maxValue*threshold)
                {
                    #if 0
                    int block_h = std::min(i+2, src.h-1) - std::max(0, i-1);
                    int block_w = std::min(j+2, src.w-1) - std::max(0, j-1);
                    if(maxValue == cornerStrength.block(std::max(0,i-1),std::max(0,i-1),block_h,block_w).maxCoeff())
                        p_dst_data[i*dst.w + j] = 255;
                    #endif
                    double temp_value = 0.0;
                    for(int m = i - 1; m < i + 2; m++)
                    {
                        for(int n = j - 1; n < j + 2; n++)
                        {
                            //抛弃超出范围的position
                            if(m>=0 && n>=0 && m < src.h && n < src.w)
                            {
                                temp_value = (cornerStrength(m,n)>temp_value) ? cornerStrength(m,n):temp_value;
                            }                                
                        }
                    }
                    if(temp_value == maxValue)
                    {
                        p_dst_data[i*dst.w + j] = 255; 
                    }
                    temp_value = 0; //update temp max value
                }
            }
            else
            {
                if(cornerStrength(i,j)>maxValue*threshold)
                    p_dst_data[i*dst.w + j] = 255;
            }
        }
    }
    
    return dst;
}

/*****************************************************************************
*   Function name: CellHistogram
*   Description  : 每个cell构建直方图
*   Parameters   : cell_m               cell中幅度矩阵8*8
*                  cell_d               cell中方向矩阵8*8
*                  bin_size             180划分成多少bin
*   Return Value : vector<double>       直方图信息(索引是x轴，值是直方图高度)
*   Spec         :
*        
*   History:
*
*       1.  Date         : 2020-3-8  10:21
*           Author       : YangLin
*           Modification : Created function
*****************************************************************************/
vector<double> CellHistogram(MatrixXd cell_m, MatrixXd cell_d, int bin_size)
{
    assert(cell_m.rows()==8 || cell_m.cols()==8); //每个cell必须是8*8像素
    assert(cell_d.rows()==8 || cell_d.cols()==8);
    int angle_unit = 20;//20度是每个bin的范围
    vector<double> cell_his(bin_size, 0);
    for(int i = 0; i < cell_m.rows(); i++)
    {
        for(int j = 0; j < cell_m.cols(); j++)
        {
            int magnitude =  cell_m(i,j); //幅度
            int angle = cell_d(i,j);      //方向
            //采用双线性插值的方式 填充直方图
            int idx = angle / angle_unit; 
            int mod = angle % angle_unit;

            cell_his[idx%bin_size] += magnitude*(1-mod/angle_unit);
            cell_his[(idx+1)%bin_size] += magnitude*(mod/angle_unit);
        }
    }
    return cell_his;
}

/*****************************************************************************
*   Function name: DetectHOGDescription
*   Description  : HOG特征提取
*   Parameters   : src                  原始图像
*                  cell_size            cell中含有多少像素(一般是8*8)
*                  bin_size             180划分成多少bin
*   Return Value : vector<vector<vector<double>>>       提取到的特征
*   Spec         :
*        
*   History:
*
*       1.  Date         : 2020-3-8  0:01
*           Author       : YangLin
*           Modification : Created function
*****************************************************************************/
vector<vector<vector<double>>> DetectHOGDescription(Image &src, int cell_size, int bin_size)
{
    assert(src.c == 1 || 180%bin_size == 0); // single channel pic, bin_size should be divisible by 180
    //Gamma校正，进行标准化
    Image src_gamma = GammaCorrection(src, 1);

    //src pic uint8转成MatrixXd
    unsigned char *p_src_data = (unsigned char *)src_gamma.data;
    MatrixXd srcMat(src.h, src.w);
    for(int i = 0; i < src.h; i++)
    {
        for(int j = 0; j < src.w; j++)
        {   
            srcMat(i,j) = p_src_data[i*src.w + j];
        }
    }

    //获取sobel算子
    MatrixXd sobX(3,3), sobY(3,3);
    GetSobel(3, sobX, sobY);

    // 计算图像I(x,y)在X和Y两个方向的梯度Ix、Iy
    MatrixXd dstX = FilterMatrix2d(srcMat, sobX);
    MatrixXd dstY = FilterMatrix2d(srcMat, sobY);    
    MatrixXd mgMat(dstX.rows(), dstX.cols()); //梯度大小(magnitude of gradient)
    MatrixXd dgMat(dstX.rows(), dstX.cols()); //梯度方向(direction of gradient)
    for(int i = 0; i < dstX.rows(); i++)
    {
        for(int j = 0; j < dstX.cols(); j++)
        {
            double magnitude = sqrt(dstX(i,j)*dstX(i,j)+dstY(i,j)*dstY(i,j));
            double direction = atan2(dstX(i,j),dstY(i,j));
            mgMat(i,j) = magnitude;
            dgMat(i,j) = (direction>180)?(direction-180):direction;
        }
    }

    //计算每个cell单元的梯度直方图
    /*说明一下cell需要多少内存， 一个cell是8*8个像素点，
    * 但是求直方图只需要关注x轴有多少个，即bin_size, 可以用长度为9的数组表示一个cell的直方图 
    */
   int cell_vec_h = src.h / cell_size;
   int cell_vec_w = src.w / cell_size;
   vector<vector<double> > cell_vec;
   for(int i = 0; i < cell_vec_h; i++)
   {
        for(int j = 0; j < cell_vec_w; j++)
        {
            //遍历每个cell
            MatrixXd cell_magnitude = mgMat.block(i*cell_size, j*cell_size, 8, 8);
            MatrixXd cell_direction = dgMat.block(i*cell_size, j*cell_size, 8, 8);
            vector<double> cell_histogram = CellHistogram(cell_magnitude, cell_direction, bin_size);
            cell_vec.push_back(cell_histogram);
        }
   }

    //生成HOG特征
    vector<vector<vector<double>>> hog_vec;
    for(int i = 0; i < cell_vec_h - 1; i++)
    {
        for(int j = 0; j < cell_vec_w - 1; j++)  
        {
            //生成block HOG特征
            vector<vector<double>> block_vec;
            block_vec.push_back(cell_vec[i*cell_vec_h+j]);
            block_vec.push_back(cell_vec[i*cell_vec_h+j+1]);
            block_vec.push_back(cell_vec[(i+1)*cell_vec_h+j]);
            block_vec.push_back(cell_vec[(i+1)*cell_vec_h+j+1]);

            //归一化block中特征
            //计算元素平方和，相当于矩阵的L2范数
            double sum_magnitude = 0.0;
            for(int m = 0; m < block_vec.size(); m++)
            {
                for(int n = 0; n < block_vec[0].size(); n++)
                {
                    sum_magnitude += pow(block_vec[m][n], 2);
                }
            }
            // 归一化
            for(int m = 0; m < block_vec.size(); m++)
            {
                for(int n = 0; n < block_vec[0].size(); n++)
                {
                    block_vec[m][n] = block_vec[m][n] / sum_magnitude;
                }
            } 
            hog_vec.push_back(block_vec);         
        }
    }

    return hog_vec;
}

} //namespace opendip