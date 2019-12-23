#ifndef ___OPENDIP_ALGORITHM_H_
#define ___OPENDIP_ALGORITHM_H_
#include "image.h"

namespace opendip {
    // 最邻近插值法(Nearest Interpolation)
    Image LinearInterpolation(Image &src_image, int resize_row, int resize_col);
    // 双线性插值法(Bilinear Interpolation)
    Image BilinearInterpolation(Image &src_image, int resize_w, int resize_h);
} 

#endif //___OPENDIP_ALGORITHM_H_