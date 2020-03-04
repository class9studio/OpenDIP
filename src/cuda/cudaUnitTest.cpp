#define CATCH_CONFIG_MAIN          //catch2µÄmainº¯Êý
#include "../../3rd_party/catch2/catch2.h"

#include "cudaCommon.h" 

TEST_CASE("cuda")
{
	cudaDeviceTest();
	REQUIRE(true);
}
