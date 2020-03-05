#define CATCH_CONFIG_MAIN          //catch, main
#include "../../3rd_party/catch2/catch2.h"

#include "cudaCommon.h" 

#if 0
TEST_CASE("cuda")
{
	cudaDeviceTest();
	REQUIRE(true);
}

TEST_CASE("cuda-vecAdd")
{
	int N = 1024;
	cudaVecAddTest(N);
	REQUIRE(true);
}
#endif

TEST_CASE("opendip-cudaStencil")
{
	int N = 30;
	cudaStencilTest(N);
	REQUIRE(true);
}
