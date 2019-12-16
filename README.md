# OpenDIP
This repo is an Open source for Digital Image Processing.

Contribute by @[kingLCH](https://github.com/kingLCH),@[jinbaoziyl](https://github.com/jinbaoziyl)

Writen by C/C++ totally, cross platform.


Instruction for each folders:

|   folders       | comment                                  | others  |
|   ------------- |:----------------------------------------:| -------:|
|   3rd_party     | for 3rd dependent lib (manually tanshou) |         |
|   data          | for test image, result image and so on   |         |
|   include	  | head file                                |         |
|   unit_tests       | demos for each function usage         |         |
|   src           | source code                              |         |
## Build System
 Cmake build this project. OpenDIP project have a requirement for including third party libraries, such as Catch2, Eigen or OpenCV.
 Environment Requirement for build:
```
$ uname -a
Linux yanglin 4.18.0-15-generic #16~18.04.1-Ubuntu SMP Thu Feb 7 14:06:04 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
```

### Catch2、Eigen Lib Support
As catch2 or Eigen3 are available as a single header file I have downloaded it and checked it into my repository, source code in document ‘3rd_party/catch2’

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
