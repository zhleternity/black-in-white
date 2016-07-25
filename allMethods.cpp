//
//  allMethods.cpp
//  bookseg
//
//  Created by brdev on 15/12/16.
//  Copyright © 2015年 brdev. All rights reserved.
//

#include "allMethods.hpp"
#include <opencv2/line_descriptor.hpp>
#include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
//#include "opencv2/legacy.hpp"
//#include <opencv2/nonfree/nonfree.hpp>
//#include "bmp.h"
//#include "MWIS.h"
#include "ScanBoardFilter.hpp"

using namespace std;
using namespace cv;


void ls::perspective(Mat &src,float in_point[8],Mat &dst)
{
    float w_a4 = sqrt(pow(in_point[0] - in_point[2], 2) + pow(in_point[1] - in_point[3] ,2 ));
    float h_a4 = sqrt(pow(in_point[0] - in_point[4], 2) + pow(in_point[1] - in_point[5] ,2));
    dst = Mat::zeros(h_a4, w_a4, CV_8UC3);
    
    //__android_log_print(ANDROID_LOG_INFO, "SRC", "error%d", src.rows);
    
    
    // corners of destination image with the sequence [tl, tr, bl, br]
    vector<Point2f> dst_pts, img_pts;
    dst_pts.push_back(Point(0, 0));
    dst_pts.push_back(Point(w_a4 - 1, 0));
    dst_pts.push_back(Point(0, h_a4 - 1));
    dst_pts.push_back(Point(w_a4 - 1, h_a4 - 1));
    
    // corners of source image with the sequence [tl, tr, bl, br]
    img_pts.push_back(Point(in_point[0], in_point[1]));
    img_pts.push_back(Point(in_point[2],in_point[3]));
    img_pts.push_back(Point(in_point[4],in_point[5]));
    img_pts.push_back(Point(in_point[6], in_point[7]));
    
    //                    __android_log_print(ANDROID_LOG_INFO, "66667777", "point=%f + %f + %f + %f + %f + %f + %f + %f", in_point[0], in_point[1] , in_point[2] ,in_point[3],in_point[4],in_point[5],in_point[6],in_point[7]);
    
    
    // convert to original image scale
    //    for (size_t i = 0; i < img_pts.size(); i++) {
    //        img_pts[i].x *= scale;
    //        img_pts[i].y *= scale;
    //    }
    
    // get transformation matrix
    Mat transmtx = getPerspectiveTransform(img_pts, dst_pts);
    
    // apply perspective transformation
    warpPerspective(src, dst, transmtx, dst.size());
}


void ILPF(Mat &src, const double D0)
{
    int i, j;
    int state = -1;
    double tempD = 0.0;
    int width, height;
    width = src.cols;
    height = src.rows;
    
    long x, y;
    x = width / 2;
    y = height / 2;
    
    //        CvMat* H_mat;
    Mat H_mat(height,width, CV_64FC2);
    for(i = 0; i < height; i++)
    {
        uchar *data_out = H_mat.ptr<uchar>(i);
        for(j = 0; j < width; j++)
        {
            if(i > y && j > x)
            {
                state = 3;
            }
            else if(i > y)
            {
                state = 1;
                
            }
            else if(j > x)
            {
                state = 2;
            }
            else
            {
                state = 0;
            }
            
            switch(state)
            {
                case 0:
                    tempD = (double)  (i * i + j * j);tempD = sqrt(tempD);break;
                case 1:
                    tempD = (double)  ((height - i) * (height - i) + j * j);tempD = sqrt(tempD);break;
                case 2:
                    tempD = (double)  (i * i + (width - j) * (width - j));tempD = sqrt(tempD);break;
                case 3:
                    tempD = (double)  ((height - i) * (height - i) + (width - j) * (width - j));tempD = sqrt(tempD);break;
                default:
                    break;
            }
            
            //二维高斯低通滤波器传递函数
            /*tempD = exp(-0.5 * pow(tempD / D0, 2));
             ((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2] = tempD;
             ((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2 + 1] = 0.0;*/
            
            //衰减系数为2的二维指数低通滤波器传递函数
            /*	tempD = exp(-pow(tempD / D0, 2));
             ((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2] = tempD;
             ((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2 + 1] = 0.0;*/
            
            //2阶巴特沃思低通滤波器传递函数
            tempD = 1 / (1 + pow(tempD / D0, 2 * 2));
            data_out[j] = tempD;
            data_out[2*j+1]  = 0.0;
            
            
            //二维理想低通滤波器传递函数
            //	if(tempD <= D0)
            //	{
            //		((double*)(H_mat->data.ptr + H_mat->step * i))[j *2] = 1.0;
            //    	//((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2 + 1] = 0.0;
            //	}
            //	else
            //	{
            //		((double*)(H_mat->data.ptr + H_mat->step * i))[j*2 ] = 0.0;
            //    	//((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2 + 1] = 0.0;
            //	}
        }
    }
    //        Mat dst(height,width,H_mat.type());
    mulSpectrums(src, H_mat, src,CV_DXT_ROWS);
    //        cvReleaseMat(&H_mat);
}

void ls::adaptiveAddFilter(Mat &src,Mat &dst)
{
    Mat image;
    resize(src,image,Size(((float)src.cols/(float)src.rows)*krows,krows));
    //        detailEnhance(image, image);
    //        edgePreservingFilter(image, image);
    //        imshow("edgepre", image);
    Scan ss;
    float gamma = ss.getGamma(image);
    cout<<gamma<<endl;
    //        if (gamma >= 0.4 && gamma < 0.5)
    //            gamma =  gamma - 0.15;
    //        else if (gamma >= 0.5 && gamma < 0.6)
    //            gamma = gamma - 0.25;
    //        else if (gamma >= 0.6)
    //            gamma = gamma - 0.35;
    ////        ss.gammaCorrection(image, image, gamma);
    Histogrom1D h1;
    double mean,var;
    h1.getMeanVar(image, mean, var);
    cout<<mean<<","<<var<<endl;
    ////        int k;
    ////        h1.changeRGB(image, k);
    //////        detailEnhance(image, image);
    ////        imshow("change", image);
    //        int mean;
    //        Mat hist = h1.getHistogramImage(image,mean);
    ////        cout<<mean<<endl;
    ////        imshow("histogram", hist);
    //        int n;
    //        if (mean <= 10 || mean > 18)
    //            n = 1.0 * mean;
    //        else if (mean > 15 && mean <= 18)
    //            n = 3.0 * mean;
    //        else
    //            n = 1.8 * mean;
    dst = image + Scalar(var,var,var);
    h1.stretch(dst, 100);
    //        detailEnhance(dst, dst);
    //        edgePreservingFilter(dst, dst);
    //        stylization(dst, dst);
    //        imshow("dst", dst);
}

void ls::adaptiveRGBFilter(Mat &src,Mat &image)
{
    //        Mat image;
    resize(src,image,Size(((float)src.cols/(float)src.rows)*krows,krows));
    //        detailEnhance(image, image);
    Histogrom1D h1;
    
    Scan ss;
    //        ss.gammaCorrection(image, image, 0.8);
    int me,kmax;
    double mean,stddev;
    h1.getMeanVar(image, mean, stddev);
    cout<<mean<<","<<stddev<<endl;
    if (mean > 128)
    {
        ss.gammaCorrection(image, image, 0.4);
    }
    else if (mean < 128)
        ss.gammaCorrection(image, image, 1.2);
    if (stddev < 30) {//对比度低
        h1.stretch(image, 100);
    }
    h1.getKValue(image, me, kmax);
    cout<<me<<","<<kmax<<endl;
    h1.changeRGB(image);
    //        detailEnhance(image, dst);
    //        morphologyEx(image, image, MORPH_OPEN, Mat(2,2,CV_8U),Point(-1,-1),1);
    //        imshow("src1", dst);
    
}

void ls::gammaSingleFilter(Mat &src,Mat &dst)
{
    Mat image;
    resize(src,image,Size(((float)src.cols/(float)src.rows)*krows,krows));
    Scan ss;
    Histogrom1D h1;
    ColorHistogram coh;
    // coh.colorReduce(img,16);
    //    h1.ScanImageAndReduceC(img, img.data);
    //    h1.ScanImageAndReduceIterator(img, img.data);
    //    h1.colorReduce0(img,84);
    float gamma = ss.getGamma(image);
    ss.gammaCorrection(image , dst, gamma);
    //    transpose(img, img);
    //    flip(img, img, 1);
    //        imshow("reduce", dst);
    
    
}


void ls::matrixAddFilter(Mat &src,Mat &dst)
{
    Mat image;
    resize(src,image,Size(((float)src.cols/(float)src.rows)*krows,krows));
    Mat re;
    Histogrom1D h1;
    Scan ss;
    re = re/3;
    re = image*1.1 + Scalar(105,105,105 );
    
    h1.stretch(re, 10);
    ss.gammaCorrection(re, re, 3);
    detailEnhance(re, dst);
    //    re = re * 2.6;
    //        imshow("ilpf", re);
    
    
}

void ls::gamma2AddFilter(Mat &src,Mat &dst)
{
    Mat image;
    resize(src,image,Size(((float)src.cols/(float)src.rows)*krows,krows));
    ColorHistogram ch;
    Histogrom1D h1;
    Scan ss;
    int mean;
    Mat dst_,dst1;
    Mat hist =  h1.getHistogramImage(image,mean);
    //        imshow("hist", hist);
    //        imwrite("/Users/brdev/Desktop/histred.jpg", hist);
    //    h1.stretch(image, 220);
    //    GaussianBlur(image, image, Size(3,3), 0);
    //        imshow("stretch", image);
    //gammaCorrection(image, image, 0.45);//0.4
    
    //    float gamma = ss.getGamma(image);
    ss.gammaCorrection(image, dst_, 0.5);//0.6
    //        imshow("0.x", dst_);
    Mat white = dst.clone();
    ss.gammaCorrection(image, dst1,2.1);//2.1
    //        imshow("1.x", dst1);
    //    dst1.copyTo(dst);
    //    imshow("copy", dst);
    //    Mat lap;
    //    Laplacian(dst, lap, dst.depth());
    //    Mat E(dst.size(), dst.type());
    //    dst1.convertTo(E, dst.type());
    Mat contrast;
    //    subtract(dst, E, contrast);
    add(dst_, dst1, contrast);
    //    addWeighted(dst, 0.1, dst1, 5, 30, contrast);
    //        imshow("add", contrast);
    h1.stretch(contrast, 0);
    //    imshow("stretch", contrast);
    Mat F(image.size(),image.type());
    contrast.convertTo(F, image.type());
    Mat result_img1,result_img2;
    //    ss.gammaCorrection(contrast, result_img1, 0.3);
    //    ss.gammaCorrection(contrast, m2, 1.5);
    detailEnhance(contrast, dst);
    
    //    h1.changeRGB(m2, 3, 0);
    //    transpose(final, final);
    //    flip(final, final, 0);
    //        imshow("image", dst);
    //        imwrite("/Users/brdev/Desktop/out4.jpg", dst);
    
    
    
}
