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

/*****************************************************************************
*   Function name: DetectOriginLBP
*   Description  : 原始LBP特征描述
*   Parameters   : src          原始图像
*   Return Value : Image        LBP纹理图
*   Spec         :
*          原始的LBP算子定义在像素3*3的邻域内，以邻域中心像素为阈值，相邻的8个像素的灰度值与邻域中心的像素值进行比较，
*       若周围像素大于中心像素值，则该像素点的位置被标记为1，否则为0。这样，3*3邻域内的8个点经过比较可产生8位二进制数，
*       将这8位二进制数依次排列形成一个二进制数字，这个二进制数字就是中心像素的LBP值; 中心像素的LBP值反映了该像素周围区域的纹理信息
*   History:
*
*       1.  Date         : 2020-3-10  10:58
*           Author       : YangLin
*           Modification : Created function
*****************************************************************************/
Image DetectOriginLBP(Image &src)
{
    assert(src.c == 1); //support gray pic only
    //Pic Map to Matrix
    GrayImgMap src_mat = GrayImgCvtMap(src);

    Image dst(src.w-2, src.h-2, 1);
    unsigned char *p_dst_data = (unsigned char *)dst.data;
    memset(p_dst_data, 0, dst.w*dst.h);
    for(int i = 1; i < src.h - 1; ++i)
    {
        for(int j = 1; j < src.w - 1; ++j)
        {
            unsigned char lbpCode = 0;
            unsigned char center = src_mat(i,j);
            lbpCode |= (src_mat(i-1,j-1) > center) << 7; //从z左上角，顺时针
            lbpCode |= (src_mat(i-1,j  ) > center) << 6;
            lbpCode |= (src_mat(i-1,j+1) > center) << 5;
            lbpCode |= (src_mat(i  ,j+1) > center) << 4;
            lbpCode |= (src_mat(i+1,j+1) > center) << 3;
            lbpCode |= (src_mat(i+1,j  ) > center) << 2;
            lbpCode |= (src_mat(i+1,j-1) > center) << 1;
            lbpCode |= (src_mat(i  ,j-1) > center) << 0; 
            p_dst_data[(i-1)*dst.w + j -1] = lbpCode;       
        }
    }

    return dst;
}

/*****************************************************************************
*   Function name: DetectCircleLBP
*   Description  : 原始LBP特征描述
*   Parameters   : src          原始图像
*                  radius       邻域半径
*                  neighbors    采样数
*   Return Value : Image        LBP纹理图
*   Spec         :
*        满足不同尺寸和频率纹理的需要。为了适应不同尺度的纹理特征，并达到灰度和旋转不变性的要求
*        将 3×3 邻域扩展到任意邻域，并用圆形邻域代替了正方形邻域，改进后的 LBP 算子允许在半径为 R 的圆形邻域内有任意多个像素点
*   对于目标像素点，映射原图的点关系式：
*   xp = xc + R*cos(2πp/P)
*   yp = yc - R*sin(2πp/P)
*     其中P是采样点, p是第几个采样点
*     得出映射关系后，通过线性插值得出目标坐标的像素值
*   线性插值:              f(0,0) f(0,1)   1-y
*     f(x, y) = [1-x, x][f(1,0) f(1,1)][  y  ]
*   History:
*
*       1.  Date         : 2020-3-10  10:58
*           Author       : YangLin
*           Modification : Created function
*****************************************************************************/
Image DetectCircleLBP(Image &src, int radius, int neighbors)
{
    assert(src.c == 1);
    //Pic Map to Matrix
    GrayImgMap src_mat = GrayImgCvtMap(src);

    Image dst(src.w-2*radius, src.h-2*radius, 1);
    unsigned char *p_dst_data = (unsigned char *)dst.data;
    memset(p_dst_data, 0, dst.w*dst.h);    

    for(int k = 0; k < neighbors; k++)
    {
        //计算采样点对于中心点坐标的偏移量rx，ry
        float rx = static_cast<float>(radius * cos(2.0 * OPENDIP_PI * k / neighbors));
        float ry = -static_cast<float>(radius * sin(2.0 * OPENDIP_PI * k / neighbors));

        //对采样点偏移量分别进行上下取整
        int x1 = static_cast<int>(floor(rx));
        int x2 = static_cast<int>(ceil(rx));
        int y1 = static_cast<int>(floor(ry));
        int y2 = static_cast<int>(ceil(ry));
        //将坐标偏移量映射到0-1之间
        float tx = rx - x1;
        float ty = ry - y1;
        //根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
        float w1 = (1-tx) * (1-ty);
        float w2 =    tx  * (1-ty);
        float w3 = (1-tx) *    ty;
        float w4 =    tx  *    ty;

        //循环处理每个像素
        for(int i = radius; i < src.h-radius; ++i)
        {
            for(int j = radius; j < src.w-radius; ++j)
            {
                //获得中心像素点的灰度值
                unsigned char center = src_mat(i,j);
                //根据双线性插值公式计算第k个采样点的灰度值
                float neighbor = src_mat(i+x1,j+y1) * w1 + src_mat(i+x1,j+y2) *w2 \
                    + src_mat(i+x2,j+y1) * w3 + src_mat(i+x2,j+y2) *w4;
                //LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                p_dst_data[(i-radius)*dst.w + j - radius] |= (neighbor > center) << (neighbors-k-1); 
            }
        }
    }

    return dst;
}

/*****************************************************************************
*   Function name: DetectRotationInvariantLBP
*   Description  : 旋转不变LBP特征描述
*   Parameters   : src          原始图像
*                  radius       邻域半径
*                  neighbors    采样数
*   Return Value : Image        LBP纹理图
*   Spec         :
*       上面的LBP特征具有灰度不变性，但还不具备旋转不变性
*       不断的旋转圆形邻域内的LBP特征，选择LBP特征值最小的作为中心像素点的LBP特征
*   History:
*
*       1.  Date         : 2020-3-10  10:58
*           Author       : YangLin
*           Modification : Created function
*****************************************************************************/
Image DetectRotationInvariantLBP(Image &src, int radius, int neighbors)
{
    assert(src.c == 1);
    //Pic Map to Matrix
    GrayImgMap src_mat = GrayImgCvtMap(src);

    Image dst(src.w-2*radius, src.h-2*radius, 1);
    unsigned char *p_dst_data = (unsigned char *)dst.data;
    memset(p_dst_data, 0, dst.w*dst.h);    

    for(int k = 0; k < neighbors; k++)
    {
        //计算采样点对于中心点坐标的偏移量rx，ry
        float rx = static_cast<float>(radius * cos(2.0 * OPENDIP_PI * k / neighbors));
        float ry = -static_cast<float>(radius * sin(2.0 * OPENDIP_PI * k / neighbors));

        //对采样点偏移量分别进行上下取整
        int x1 = static_cast<int>(floor(rx));
        int x2 = static_cast<int>(ceil(rx));
        int y1 = static_cast<int>(floor(ry));
        int y2 = static_cast<int>(ceil(ry));
        //将坐标偏移量映射到0-1之间
        float tx = rx - x1;
        float ty = ry - y1;
        //根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
        float w1 = (1-tx) * (1-ty);
        float w2 =    tx  * (1-ty);
        float w3 = (1-tx) *    ty;
        float w4 =    tx  *    ty;

        //循环处理每个像素
        for(int i = radius; i < src.h-radius; ++i)
        {
            for(int j = radius; j < src.w-radius; ++j)
            {
                //获得中心像素点的灰度值
                unsigned char center = src_mat(i,j);
                //根据双线性插值公式计算第k个采样点的灰度值
                float neighbor = src_mat(i+x1,j+y1) * w1 + src_mat(i+x1,j+y2) *w2 \
                    + src_mat(i+x2,j+y1) * w3 + src_mat(i+x2,j+y2) *w4;
                //LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                p_dst_data[(i-radius)*dst.w + j - radius] |= (neighbor > center) << (neighbors-k-1); 
            }
        }
    }

    //进行旋转不变处理
    for(int i = 0; i < dst.h; ++i)
    {
        for(int j = 0; j < dst.w; ++j)
        {
            unsigned char currentValue = p_dst_data[i*src.w+j];
            unsigned char minValue = currentValue;
            for(int k = 1; k < neighbors; ++k)
            {
                //循环左移
                unsigned char temp = (currentValue>>(neighbors-k)) | (currentValue<<k);
                if(temp < minValue)
                {
                    minValue = temp;
                }
            }
            p_dst_data[i*src.w+j] = minValue;
        }
    }

    return dst;
}

} //namespace opendip