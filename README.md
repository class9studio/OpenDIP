# OpenDIP
This repo is an Open source for Digital Image Processing.

Contribute by @[kingLCH](https://github.com/kingLCH),@[jinbaoziyl](https://github.com/jinbaoziyl)

Writen by C/C++ totally, cross platform.


Instruction for each folders:

|   folders       | comment                                  | others  |
|   ------------- |:----------------------------------------:| -------:|
|   data          | for test image, result image and so on   |         |
|   include	  | head file                                |         |
|   samples       | demos for each function usage            |         |
|   src           | source code                              |         |
## Build System
 Cmake build this project. OpenDIP project have a requirement for including third party libraries, such as Catch2, Eigen or OpenCV.
 Environment Requirement for build:
```
$ uname -a
Linux yanglin 4.18.0-15-generic #16~18.04.1-Ubuntu SMP Thu Feb 7 14:06:04 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
```

### Catch2 Unit Testing Lib Support
As catch2 is available as a single header file I have downloaded it and checked it into my repository, source code in document ‘3rd_party/catch2’

### Eigen Lib Support
Eigen support by command
```
sudo apt-get install libeigen3-dev
```

Eigen support by source code
```
sudo wget https://github.com/eigenteam/eigen-git-mirror/archive/3.3.5.tar.gz
sudo tar -xzvf 3.3.5.tar.gz 
sudo mv eigen-git-mirror-3.3.5/ eigen-3.3.5/ 
cd eigen-3.3.5/ 
mkdir build 
cd build
sudo cmake .. 
sudo make 
sudo make install 
sudo ldconfig -v
```
 ### OpenCv Support
 dependencies install:
 ```
 $ sudo apt-get install build-essential libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
 ```

 download the source file
 ```
 Official website  https://opencv.org/
 github   https://github.com/opencv/opencv
 ```

 install
 ```
 $ unzip OpenCV-4.1.1.zip -d .
 $ cd ./OpenCV-4.1.1
 $ mkdir build
 $ cd ./build
 $ cmake -D CMAKE_BUILD_TYPE=Release -D OPENCV_GENERATE_PKGCONFIG=ON -D    CMAKE_INSTALL_PREFIX=/usr/local ..
 $ sudo make -j4
 $ sudo make install
 ```


 ### OpenDIP Lib Build,And Test
 ```
 $ mkdir build
 $ cd build
 $ cmake ..
 $ make
 $ ./OpenDIP

 unittest command:
$ make test
 ```
