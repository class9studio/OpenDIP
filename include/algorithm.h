#ifndef ___OPENDIP_ALGORITHM_H_
#define ___OPENDIP_ALGORITHM_H_
#include "image.h"

namespace opendip {

    Image LinearInterpolation(Image &src_image, size_t resize_row, size_t resize_col);

} 

#endif //___OPENDIP_ALGORITHM_H_