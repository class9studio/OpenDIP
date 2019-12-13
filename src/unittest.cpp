#include <iostream>
#include "unittest.h"

void UnitTest_Image_GetImageTypeFromFile(char *filename)
{
    std::cout << "image type: " << GetImageTypeFromFile(filename) << std::endl;
}