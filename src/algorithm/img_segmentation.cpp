#include <iostream>
#include <cmath>

#include "common.h"
#include "algorithm.h"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
namespace opendip {
    /*****************************************************************************
    *   Function name: GetOstu
    *   Description  : OSTU（大津算法）
    *   Parameters   : src              source image 
    *
    *   Return Value : 阈值灰度
    *   Spec         : 
    *   History:
    *
    *       1.  Date         : 2019-12-30
    *           Author       : YangLin
    *           Modification : function draft
    *****************************************************************************/
    unsigned char GetOstu(Image &src)
    {
        unsigned int *histogram = new unsigned int[256]();
        unsigned char *p_src_data = (unsigned char *)src.data;  
        for(size_t j = 0; j < src.h; j++)
        {
            for(size_t i = 0; i < src.w; i++)
            {
                histogram[p_src_data[j * src.c * src.w + src.c*i]] += 1;
            }
        }
        long size = src.w * src.h;
        unsigned char threshold;      
        long sum0 = 0, sum1 = 0; //存储前景的灰度总和及背景灰度总和  
        long cnt0 = 0, cnt1 = 0; //前景的总个数及背景的总个数  
        double w0 = 0, w1 = 0; //前景及背景所占整幅图像的比例  
        double u0 = 0, u1 = 0;  //前景及背景的平均灰度  
        double variance = 0; //最大类间方差
        double maxVariance = 0;  
        for(int i = 1; i < 256; i++) //一次遍历每个像素  
        {    
            sum0 = 0;  
            sum1 = 0;   
            cnt0 = 0;  
            cnt1 = 0;  
            w0 = 0;  
            w1 = 0;  
            for(int j = 0; j < i; j++)  
            {  
                cnt0 += histogram[j];  
                sum0 += j * histogram[j];  
            }  
    
            u0 = (double)sum0 /  cnt0;   
            w0 = (double)cnt0 / size;  
    
            for(int j = i ; j <= 255; j++)  
            {  
                cnt1 += histogram[j];  
                sum1 += j * histogram[j];  
            }  
    
            u1 = (double)sum1 / cnt1;  
            w1 = 1 - w0; // (double)cnt1 / size;  
    
            variance =  w0 * w1 *  (u0 - u1) * (u0 - u1);  
            if(variance > maxVariance)   
            {    
                maxVariance = variance;    
                threshold = i;    
            }   
        }    

        return threshold;
    }

    /*****************************************************************************
    *   Function name: EdgeDetection
    *   Description  : 单方向边缘检测
    *   Parameters   : src                  Source image name
    *                  kernel               边缘检测滤波器
    *   由于图像是离散的信号，我们可以用临近的两个像素差值来表示像素灰度值函数的导数
    *   df(x,y)/dx = (f(x,y) - f(x-1,y)) / 2
    *   譬如:
    *       x方向滤波器 [1, 0 , -1] 或者 [1, -1]
    *       y方向滤波器 [1, 0 , -1]T
    *       45°梯度方向:
    *             XY = [ 1 ,  0,           YX = [ 0 , 1,
    *                    0 , -1,                  -1, 0, 
    *                  ]                        ]
    *   另外需要注意:  经过卷积计算得像素值可能是负，需要求取绝对值
    * 
    *   Return Value : Image Type.         边缘检测输出图像
    * 
    *   Spec         :
    *   History:
    *
    *       1.  Date         : 2020-1-17
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    Image EdgeDetection(Image &src, MatrixXd &kernel)
    {
        if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 1 || 0 == kernel.size())
        {
            cout << "source image invalid" << endl;
            return Image();
        }

        return Filter2D(src, kernel);
    }
    /*****************************************************************************
    *   Function name: EdgeDetection
    *   Description  : 整幅图像的边缘检测
    *   Parameters   : src                   Source image name
    *                  kernelX               X方向边缘检测滤波器
    *                  kernelY               Y方向边缘检测滤波器
    *   Return Value : Image Type.           边缘检测输出图像
    * 
    *   Spec         :
    *                 X、Y方向得边缘滤波结果，叠加得到整幅图像得滤波结果
    *   History:
    *
    *       1.  Date         : 2020-1-17
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    Image EdgeDetection(Image &src, MatrixXd &kernelX, MatrixXd &kernelY) 
    {
        if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 1)
        {
            cout << "source image invalid" << endl;
            return Image();
        }  

        Image dstX = Filter2D(src, kernelX);
        Image dstY = Filter2D(src, kernelY);

        Image dst(src.w, src.h, src.c);
        unsigned char *p_srcX_data = (unsigned char *)dstX.data;
        unsigned char *p_srcY_data = (unsigned char *)dstY.data;
        unsigned char *p_dst_data = (unsigned char *)dst.data;
        int  value = 0;
        for(int j = 0; j < src.h; j++)
        {
            for(int i = 0; i < src.w; i++)
            {
                for(int z = 0; z < src.c; z++)
                {
                    value = p_srcX_data[j*dstX.w*dstX.c + i*dstX.c] + p_srcY_data[j*dstY.w*dstY.c + i*dstY.c];
                    p_dst_data[j*dst.w*dst.c + i*dst.c] = (value > 255 ? 255 : value);
                }
            }
        }

        return dst;
    }

    /*****************************************************************************
    *   Function name: GetSobel
    *   Description  : Sobel算子构造
    *   Parameters   : n                     Sobel算子维度
    *   Return Value : Image Type.           Sobel算子
    * 
    *   Spec         :
    *   History:
    *
    *       1.  Date         : 2020-1-17
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    static int Factorial(int num)
    {
        if(num == 0)
            return 1;
        else
            return num*Factorial(num - 1);
    }
    void GetSobel(int n, MatrixXd &sobX, MatrixXd &sobY)
    {
        int value = 0;
        MatrixXd sob(n, n);

        //先求第一列
        VectorXd sob_col(n);
        for(int  i = 0; i < n; i++)
        {
            value = Factorial(n - 1) / (Factorial(i)*Factorial(n -1 - i));
            sob_col(i) = value;
        }

        //再求第一行
        VectorXd sob_row(n);
        for(int i = 0; i < n; i++)
        {
            value = Factorial(n - 2) * (n - 1 - 2*i) / (Factorial(i)*Factorial(n -1 - i));
            sob_row(i) = value;
        }

        sobX = sob_col*sob_row.transpose();
        sobY = sob_row*sob_col.transpose();
    }

    /*****************************************************************************
    *   Function name: Sobel
    *   Description  : Sobel算子-边缘检测
    *   Parameters   : src                   输入原始图像
    *                  ksize                 Sobel算子维度n*n
    *   Return Value : Image Type.           输出检测图像
    * 
    *   Spec         :
    *         Sobel算子是一阶的梯度算子,作用: 在边缘检测的同时，对噪声具有平滑作用;
    *         3x3 sobel算子: [1, 0, -1] * [1, 2, 1]T
    *         其中:  [1, 0, -1]  ----边缘检测算子
    *                [1, 2, 1]T  ----标准平滑算子
    *             所以: Sobel具有平滑和微分的功效 
    *   History:
    *
    *       1.  Date         : 2020-1-17
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    Image Sobel(Image &src, int ksize)
    {
        if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 1 || ksize < 0)
        {
            cout << "source image invalid." << endl;
            return Image();
        }
        //获取sobel算子
        MatrixXd sobX(ksize,ksize), sobY(ksize,ksize);
        GetSobel(ksize, sobX, sobY);

        Image dstX = Filter2D(src, sobX);
        Image dstY = Filter2D(src, sobY);

        Image dst(src.w, src.h, src.c);
        unsigned char *p_srcX_data = (unsigned char *)dstX.data;
        unsigned char *p_srcY_data = (unsigned char *)dstY.data;
        unsigned char *p_dst_data = (unsigned char *)dst.data;
        int  value = 0;
        for(int j = 0; j < src.h; j++)
        {
            for(int i = 0; i < src.w; i++)
            {
                for(int z = 0; z < src.c; z++)
                {
                    value = p_srcX_data[j*dstX.w*dstX.c + i*dstX.c] + p_srcY_data[j*dstY.w*dstY.c + i*dstY.c];
                    p_dst_data[j*dst.w*dst.c + i*dst.c] = (value > 255 ? 255 : value);
                }
            }
        }

        return dst;
    }

    /*****************************************************************************
    *   Function name: Scharr
    *   Description  : Scharr算子
    *   Parameters   : src                   输入原始图像
    *   Return Value : Image Type.           输出图像
    * 
    *   Spec         :
    *                Scharr算子是对Sobel算子差异性的增强,通过将滤波器中的权重系数放大来增大像素值间的差异
    *                X:  [                                          Y:  [
    *                       -3, 0, 3,                                       -3, -10, -3,
    *                       -10,0,10,                                        0,  0 , 0,
    *                       -3, 0, 3,                                        3,  10, 3,
    *                    ]                                              ]
    * 
    *   History:
    *
    *       1.  Date         : 2020-1-18
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/
    Image Scharr(Image &src)
    {
        if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 1)
        {
            cout << "source image invalid." << endl;
            return Image();
        }
        MatrixXd Scharr_mX(3,3);
        Scharr_mX <<  -3, 0 , 3,
                      -10,0 ,10,
                      -3, 0 , 3;
        MatrixXd Scharr_mY(3,3);
        Scharr_mY <<  -3, -10 , -3,
                       0,  0 ,  0,
                       3,  10 , 3;  
                          
        Image dstX = Filter2D(src, Scharr_mX);
        Image dstY = Filter2D(src, Scharr_mY);

        Image dst(src.w, src.h, src.c);
        unsigned char *p_srcX_data = (unsigned char *)dstX.data;
        unsigned char *p_srcY_data = (unsigned char *)dstY.data;
        unsigned char *p_dst_data = (unsigned char *)dst.data;
        int  value = 0;
        for(int j = 0; j < src.h; j++)
        {
            for(int i = 0; i < src.w; i++)
            {
                for(int z = 0; z < src.c; z++)
                {
                    value = p_srcX_data[j*dstX.w*dstX.c + i*dstX.c] + p_srcY_data[j*dstY.w*dstY.c + i*dstY.c];
                    p_dst_data[j*dst.w*dst.c + i*dst.c] = (value > 255 ? 255 : value);
                }
            }
        }

        return dst;
    }

    /*****************************************************************************
    *   Function name: Laplacian
    *   Description  : Laplacian算子
    *   Parameters   : src                   输入原始图像
    *   Return Value : Image Type.           输出图像
    * 
    *   Spec         :
    *        Laplacian算子是一种二阶导数算子，对噪声比较敏感，因此常需要配合高斯滤波一起使用
    *        一阶微分:  df(x,y)/dx = f(x) - f(x -1)
    *        二阶微分:  d2f(x,y)/d2x = f(x+1) + f(x-1)-2f(x)
    *                  d2f(x,y)/d2x + d2f(x,y)/d2y = f(x, y-1) + f(x, y+1) + f(x-1,y) + f(x+1,y) -4f(x,y)
    * 
    *   History:
    *
    *       1.  Date         : 2020-1-19
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/    
    Image Laplacian(Image &src)
    {
        if(src.data == NULL || src.w < 1 || src.h < 1 || src.c < 1)   
        {
            cout << "source image invalid" << endl;
            return Image();
        }

        MatrixXd lap_m(3, 3);
        lap_m << 0,  1, 0,
                 1, -4, 1, 
                 0, 1,  0;
        return Filter2D(src, lap_m);
    }

    /*
    *  梯度方向常取值0°、45°、90°和135°这个四个角度
    *   四个角度对应编号
    *   0 1 2
    *   3 * 5
    *   6 7 8
    *   edgedirction
    */
    void GetEdgeDirection(double *edgedirection,double *sample_direction,int width,int height)
    {
        double angle=0.0;
        for(int i=0;i<width*height;i++){
            angle=edgedirection[i];
            if(angle<22.5||angle>=337.5)
                sample_direction[i]=5.0;
            else if(angle<67.5&&angle>=22.5)
                sample_direction[i]=2.0;
            else if(angle<112.5&&angle>=67.5)
                sample_direction[i]=1.0;
            else if(angle<157.5&&angle>=112.5)
                sample_direction[i]=0.0;
            else if(angle<202.5&&angle>=157.5)
                sample_direction[i]=3.0;
            else if(angle<247.5&&angle>=202.5)
                sample_direction[i]=6.0;
            else if(angle<292.5&&angle>=247.5)
                sample_direction[i]=7.0;
            else if(angle<337.5&&angle>=292.5)
                sample_direction[i]=8.0;
            else if(angle==-1.0)
                sample_direction[i]=-1.0;
            }
    }

    void Non_MaxSuppression(double *src,double *dst,double *dirction,int width,int height)
    {
        double *temp=(double*)malloc(sizeof(double)*width*height);
        int dir;
        int y;
        int x;
        double value_c;
        memset(temp, 0, width*height);
        
        for(int j = 1; j < height-1; j++)
        {
            for(int i = 1; i < width-1; i++)
            {
                if(dirction[j*width+i]!=-1.0){
                    dir=(int)dirction[j*width+i];
                    y=dir/3-1;
                    x=dir%3-1;
                    value_c=src[j*width+i];
                    if(value_c<=src[(j+y)*width+i+x]||value_c<src[(j-y)*width+i-x])
                        temp[j*width+i]=0.0;
                    else
                        temp[j*width+i]=value_c;
                }
            }
        }
        for(int i = 0; i < width*height; i++)
            dst[i] = temp[i];
        free(temp);
    }

    void EdgeTrack(double *src, int width, int height, Point *seed)
    {
        int x=seed->x;
        int y=seed->y;
        if(x>=0&&x<width&&y>=0&&y<height&&src[y*width+x]==1.0)
        {
            src[y*width+x]=2;
            for(int j=-1;j<2;j++)
            {
                for(int i=-1;i<2;i++)
                {
                    if(!(j==0&&i==0))
                    {
                        Point seed_next;
                        seed_next.x=x+i;
                        seed_next.y=y+j;
                        EdgeTrack(src,width,height,&seed_next);
                    }
                }
            }
        }
    }

    /*****************************************************************************
    *   Function name: Canny
    *   Description  : Canny算法
    *   Parameters   : src                   输入原始图像
    *                  sobel_size            sobel核大小
    *                  threshold1            第一个滞后阈值
    *                  threshold1            第一个滞后阈值
    *   Return Value : Image Type.           输出图像
    * 
    *   Spec         :
    *       Canny算法不容易受到噪声的影响，能够识别图像中的弱边缘和强边缘，并结合强弱边缘的位置关系，
    *       Canny边缘检测算法是目前最优越的边缘检测算法之一
    * 
    *   History:
    *
    *       1.  Date         : 2020-2-23
    *           Author       : YangLin
    *           Modification : Created function
    *****************************************************************************/     
    Image Canny(Image &src, int sobel_size, double threshold1, double threshold2)
    {
        assert(src.c == 1); //单通道灰度图像
        int width  = src.w;
        int height = src.h;
        double *temp = new double[sizeof(double)*width*height]();
        double *edge_a = new double[sizeof(double)*width*height]();//边缘幅度
        double *edge_d = new double[sizeof(double)*width*height]();//边缘方向
        double *threshold_max = new double[sizeof(double)*width*height]();
        double *threshold_min = new double[sizeof(double)*width*height]();

        Image dst(src.w, src.h, src.c);
        unsigned char *p_dst_data = (unsigned char *)dst.data;

        // Step1: 使用高斯滤波平滑图像，减少图像中噪声
        MatrixXd gaussian_kernel(5,5);
        gaussian_kernel <<    2, 4, 5, 4, 2,
                              4, 9,12, 9, 4,
                              5,12,15,12, 5,
                              4, 9,12, 9, 4,
                              2, 4, 5, 4, 2;
        gaussian_kernel = gaussian_kernel/139;
        Image src_blur = Filter2D(src, gaussian_kernel); 

        // Step2：计算图像中每个像素的梯度方向和幅值
        // 获取sobel算子
        MatrixXd sobX(sobel_size,sobel_size), sobY(sobel_size,sobel_size);
        GetSobel(sobel_size, sobX, sobY);

        Image dstX = Filter2D(src, sobX);
        Image dstY = Filter2D(src, sobY);
        unsigned char *dst_x = (unsigned char *)dstX.data;
        unsigned char *dst_y = (unsigned char *)dstY.data;

        double x = 0, y = 0;
        // 通过Sobel算子分别检测图像X方向的边缘和Y方向的边缘
        for(int i = 0; i < width*height; i++)
        {
            x = dst_x[i];
            y = dst_y[i];
            if(!(x==0.0 && y==0.0))
            {
                double v=atan2(y, x)*180.0/OPENDIP_PI;
                double w=atan2((abs(x*x)+abs(y*y)), 1);
                if(v<0.0)
                    v+=360.0;
                edge_d[i]=v;
                if(w > 256)
                    w=256;
                edge_a[i]=w;
            }
            else
            {
                edge_a[i]=0;
                edge_d[i]=-1.0;
            }
        }

        // Step3：应用非极大值抑制算法消除边缘检测带来的杂散响应
        GetEdgeDirection(edge_d, edge_d, width, height);
        Non_MaxSuppression(edge_a, temp, edge_d, width, height);

        // Step4：应用双阈值法划分强边缘和弱边缘
        for(int i = 0;i < width*height; i++)
        {
            threshold_max[i] = temp[i]>threshold1?temp[i]:0;
            threshold_min[i] = temp[i]>threshold2?temp[i]:0;
        }
        for(int j = 0;j < width*height; j++)
        {
            threshold_max[j]=threshold_max[j]!=0.0?1.0:0.0;
        }

        for(int j = 0; j < height; j++)
        {
            for(int i = 0; i < width; i++)
            {
                if(threshold_max[j*width+i]==1.0&&threshold_min[j*width+i]!=2.0){
                    Point p;
                    p.x=i;
                    p.y=j;
                    EdgeTrack(threshold_min, width, height, &p);
                }
            }
        }        
        
        // Step5: result
        memset(p_dst_data, 0, width*height);
        for(int i = 0; i <width*height; i++)
        {
            if(threshold_min[i]==2.0)
                dst[i]=255.0;
        }

        delete[] temp;
        delete[] threshold_max;
        delete[] threshold_min;
        delete[] edge_d;
        delete[] edge_a;
        return dst;
    }

} // namespce opendip