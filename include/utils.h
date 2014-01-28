//
//  utils.h
//  ALIEN
//
//  Author: alantrrs
//  Modified by Edgar Reyna on 11/15/13.
//  Copyright (c) 2013 Edgar Reyna. All rights reserved.

#include <opencv2/opencv.hpp>
#include <iostream>
#pragma once

// ******************Methods*******************
int factorial(int x);
int nCr(int n, int r);
float median(std::vector<float> m);
double median(std::vector<double> m);
float average(std::vector<float> av);
cv::Mat createM(float x_, float y_, float rot);
cv::Point2f getULPoint(cv::RotatedRect rbox);
std::vector<std::vector<int>> combinations(int size);
std::vector<int> index_shuffle(int begin,int end);
bool is_inside(const cv::Point pt,const cv::Rect& box);
bool is_inside(const cv::Point2f pt,const cv::RotatedRect& box);
void drawBox(cv::Mat& image, cv::Rect box, cv::Scalar color = cvScalarAll(255), int thick=1);
void drawBox(cv::Mat& image, cv::RotatedRect box, cv::Scalar color = cvScalarAll(255), int thick=1);
void displayKeypoints(cv::Mat& display, const cv::vector<cv::KeyPoint>& kpts);
std::vector<cv::DMatch> goodMatches(const std::vector<std::vector<cv::DMatch>>& kmatches, float threshold);
void showPoints(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::Point2f>& queryPts, std::vector<cv::Point2f>& modelPts, std::vector<cv::DMatch>& matches);
