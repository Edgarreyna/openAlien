//
//  Context.h
//  ALIEN
//
//  Created by Edgar Reyna on 12/6/13.
//  Copyright (c) 2013 Edgar Reyna. All rights reserved.
//

#ifndef __ALIEN__Context__
#define __ALIEN__Context__

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>

class Context{
private:
    cv::Mat occ_descriptors;
    cv::Mat descriptors;    //time Window
    std::vector<int> time;
    std::vector<int> oversample;
    void timeWindow();
    
public:
    void save(cv::FileStorage& file);
    void init(cv::Mat& descriptors_);
    void update(cv::Ptr<cv::DescriptorMatcher> matcher, cv::Mat& descriptors_, cv::Mat& occ_);
    cv::Mat getDescriptors(){ return descriptors;}
    
};
#endif /* defined(__ALIEN__Context__) */
