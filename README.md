# OpenDIP
## 简介
This repo is an Open source for Digital Image Processing.

Contribute by @[kingLCH](https://github.com/kingLCH),@[jinbaoziyl](https://github.com/jinbaoziyl)

Writen by C/C++ totally, cross platform.


Instruction for each folders:

|   folders       | comment                                  | others  |
|   ------------- |:----------------------------------------:| -------:|
|   3rd_party     | for 3rd dependent lib (manually tanshou) |         |
|   data          | for test image, result image and so on   |         |
|   include	      | head file                                |         |
|   unit_tests    | demos for each function usage            |         |
|   src           | source code                              |         |
|   docs          | documents                                |         |

## Build System
 Cmake build this project. OpenDIP project have a requirement for including third party libraries, such as Catch2, Eigen or OpenCV.
 Environment Requirement for build:
```
$ uname -a
Linux yanglin 4.18.0-15-generic #16~18.04.1-Ubuntu SMP Thu Feb 7 14:06:04 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
```

### Catch2、Eigen Lib Support
As catch2 or Eigen3 are available as a single header file I have downloaded it and checked it into my repository, source code in document ‘3rd_party/catch2’

### Options Lib Support
Opencv,Matplotlibcpp are optional in Root Cmakelist.txt, as follows:
```
option(WITH_OPENCV "With OpenCV Support" ON)
option(WITH_MATPLOTLIB_CPP "With Matplotlib-cpp Support" ON)
```
Notice: To support Matplotlibcpp, install software in Ubuntu:
```
sudo apt-get install python-matplotlib python-numpy python2.7-dev
```

### OpenDIP Lib Build,And Test
 ```
 $ mkdir build
 $ cd build
 $ cmake ..
 $ make

 $ ./OpenDIP
 $ make test
 or
 $ ./unit_tests
 ```

 ## 目录
 * [简介](README.md)
 * [1.Build System](README.md)
   * [1.1 Catch2、Eigen Lib Support](README.md)
   * [2.2 Options Lib Support](README.md)
   * [3.3 OpenDIP Lib Build,And Test](README.md)
 * [2.Opendip](include/algorithm.h)
   * [2.1 OpenDip基础](docs/OpenDip基础.md)
   * [2.2 数字图像基础](docs/数字图像基础.md)
   * [2.3 灰度变换](docs/灰度变换.md)
   * [2.4 空间滤波](docs/空间滤波.md)
   * [2.5 频率滤波](docs/频域滤波.md)
   * [2.6 形态学图像处理](docs/图像图形学.md)
   * [2.7 图像分割](docs/图像分割.md)
   * [2.8 特征提取](docs/特征提取.md)

持续更新中......

