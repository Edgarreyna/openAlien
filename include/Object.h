//
//  Object.h
//  ALIEN
//
//  Created by Edgar Reyna on 1/17/14.
//  Copyright (c) 2014 Edgar Reyna. All rights reserved.
//

#ifndef __ALIEN__Object__
#define __ALIEN__Object__

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include "utils.h"

//*******************STRUCTURES********************

struct Model{
    cv::Mat patch;
    std::vector<cv::Point2f> keypoints;
    cv::Mat descriptors;
    cv::RotatedRect bb;
};

class Object{
private:
    cv::Mat descriptors;    //Collects object descriptors indefinitely
    cv::Mat init_patch;
    Model obj;              // Object template
    
public:
    void save(cv::FileStorage& file);
    void init(const cv::Mat& img, cv::Mat& descriptors_, std::vector<cv::Point2f>& keypts, cv::RotatedRect rbox);
    void update(cv::Ptr<cv::DescriptorMatcher>& matcher, cv::Mat& descriptors_, std::vector<cv::Point2f>& keypts);
    void getTemplate(cv::Mat& descriptors, std::vector<cv::Point2f>& keypts);
    cv::Mat getDescriptors(){ return descriptors;}
};
#endif /* defined(__ALIEN__Object__) */
