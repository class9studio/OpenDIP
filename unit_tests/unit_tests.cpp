#include <iostream>

#include "common.h"
#include "image.h"

#define CATCH_CONFIG_MAIN          //catch2的main函数
#include "catch2.h"


TEST_CASE( "simple" )
{
    REQUIRE( OPENDIP_IMAGE_PNG == GetImageTypeFromFile((char *)"yanglin.png") );
}
