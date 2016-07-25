//
//  ScanBoardFilter.hpp
//  bookseg
//
//  Created by brdev on 15/12/4.
//  Copyright © 2015年 brdev. All rights reserved.
//

#ifndef ScanBoardFilter_hpp
#define ScanBoardFilter_hpp

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class Scan{
public:
    void calhistOfRGB(Mat& src);
    float getGamma(Mat &src);
    void gammaCorrection(Mat &src, Mat &dst, float fGamma);
};

#endif /* ScanBoardFilter_hpp */
