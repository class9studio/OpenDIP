#include <iostream>
#include <cmath>

#include "common.h"
#include "algorithm.h"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
namespace opendip {
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

        //将图像map到Matrix
        MapType dstX_m = ImageCvtMap(dstX);
        MapType dstY_m = ImageCvtMap(dstY);
        dstX_m = dstX_m + dstY_m;
        return dstX;
    }

} // namespce opendip