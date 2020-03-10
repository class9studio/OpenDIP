#define CATCH_CONFIG_MAIN          //catch, main
#include "catch2.h"
#include "common.h"
#include "image.h"
#include "timing.h"
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

TEST_CASE("opendip-cudaStencil")
{
	int N = 30;
	cudaStencilTest(N);
	REQUIRE(true);
}
#endif

TEST_CASE("opendip-cudaRGB2Gray")
{
	Image src = ImgRead("../../../data/test_image/lena.jpg");
	double startTime = now();
	Image dst = cudaOpenDipRGB2Gray(src);
	double nDetectTime = calcElapsed(startTime, now());
	printf("LinearInterpolation time: %d ms.\n ", (int)(nDetectTime * 1000));

	ImgWrite("../../../data/output_image/linux/lena_cuda.jpg", dst);
	REQUIRE(true);
}
