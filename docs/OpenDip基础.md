# 2.1 OpenDip基础
OpenDip是用于学习数字图像处理的开源项目。本节简单介绍本项目引用的一些软件。

## 2.1.1 [Catch2](https://github.com/catchorg/Catch2)
Catch2用于本项目单元测试，对每次更新的图像算法都需要单元测试代码，测试功能正确后才能提交。
Catch2是一个header-only的开源库，只需要在项目CmakeLists.txt文件中指定头文件就可以使用。
```
# add the CMakeFile that defines catch2
add_subdirectory(3rd_party/catch2)
```

## 2.1.2 Eigen3
Eigen3用于支持本项目所需的线性代数，矩阵和矢量运算，数值分析及其相关的算法
Eigen3是C++ template library，使用Eigen3库，只需包特定模块的的头文件即可; 如果已经安装eigen3那么使用find_package寻找库信息，
然后添加到项目即可；或者下载源代码到本项目，然后直接执行头文件目录
```
# include third package library, then check
find_package (Eigen3  REQUIRED NO_MODULE)
if(Eigen3_FOUND)
    message ("Eigen3 found")
	set(EIGEN3_INCLUDE_DIR ${EIGEN_INCLUDE_DIRS})
else()
    message (FATAL_ERROR "Cannot find Eigen3, then install")
	add_subdirectory(3rd_party/eigen3)
	set(EIGEN3_INCLUDE_DIR ${EIGEN_INCLUDE_DIRS})
endif()
```
## 2.1.3 OpenCV
OpenCV是众所周知的图像处理开源项目，涵盖几乎所有的图像算法。
本项目支持opencv的目的是，有时候需要对比两者的效果、或者运行速率。
```
Notice:  
    Opendip项目中的算法绝对不允许直接拿opencv的函数来调用。
```

OpenCV是可选择支持，在CmakeLists.txt文件中指定开关
```
# Set Options
option(WITH_OPENCV "With OpenCV Support" ON)
```
支持opencv编译:
```
if(WITH_OPENCV)
    find_package(OpenCV REQUIRED)
    if(OpenCV_FOUND)
        message("OpenCV found")
        target_link_libraries(OpenDIP
        ${OpenCV_LIBS}) 
        target_include_directories(OpenDIP PUBLIC 
            ${OpenCV_INCLUDE_DIRS}) 
    else()
        message(FATAL_ERROR "cannot find OpenCV")
    endif()
endif()
```

## 2.1.4 Matplotlib-cpp
matplotlib-cpp用于本项目高效的图表绘制与数据可视化，它是遵循Matlab and matplotlib接口设计的.
Extremely simple yet powerful header-only C++ plotting library built on the popular matplotlib

