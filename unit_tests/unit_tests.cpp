#include <iostream>
#include <Eigen/Dense>
#include "common.h"
#include "image.h"

#define CATCH_CONFIG_MAIN          //catch2的main函数
#include "catch2.h"
using namespace Eigen;

TEST_CASE( "simple" )
{
    REQUIRE( OPENDIP_IMAGE_PNG == GetImageTypeFromFile((char *)"yanglin.png") );
}

TEST_CASE("empty")
{       
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << "Here is the matrix m:\n" << m << std::endl;
  VectorXd v(2);
  v(0) = 4;
  v(1) = v(0) - 1;
  std::cout << "Here is the vector v:\n" << v << std::endl;

 REQUIRE( 0 == 0 );
}
