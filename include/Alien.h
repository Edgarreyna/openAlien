//
//  Alien.h
//  ALIEN
//
//  Created by Edgar Reyna on 11/7/13.
//  Copyright (c) 2013 Edgar Reyna. All rights reserved.
//

#ifndef __ALIEN__Alien__
#define __ALIEN__Alien__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <math.h>
#include "Object.h"
#include "Context.h"
#include "utils.h"


class Alien{
private:
    //Parameters
    cv::FileNode params_;
    int min_win;
    int patch_size;
    cv::Rect init_box_;
    cv::Point2d frameSize;
    const double PI = std::atan(1.0)*4;
    
    //Last frame data
    cv::Mat lastM;
    cv::RotatedRect lastbox;
    cv::Rect lastArea;
    cv::Point2f lastDisp;
    float lastRot;
    bool lastvalid;

    //Current frame data
    bool validM;
    int framesLost;

    // Features Extractor
    cv::SIFT featuresExtractor;
    
    // Object Model
    Object object;
    
    // Context Model
    Context context;
    
    // Matcher
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    //cv::FlannBasedMatcher matcher_;
    
    // Methods
    void read(const cv::FileNode& file);
    cv::Rect searchArea(const cv::Rect &box);
    bool checkM(cv::Mat& M);
    cv::Mat removeFeatures(const cv::Mat& descriptors,std::vector<cv::KeyPoint>& keypts, std::vector<cv::DMatch>& obj_matches, std::vector<cv::DMatch>& con_matches);
    cv::Mat homography(const std::vector<cv::Point2f>& queryPts, const std::vector<cv::Point2f>& modelPts, std::vector<cv::DMatch>& matches);
    float mlesac(const std::vector<cv::Point2f>& queryPts, const std::vector<cv::Point2f>& modelPts, std::vector<cv::DMatch>& matches, cv::Mat& best_M);
    void newBox(const cv::Mat& M, cv::RotatedRect& rbox);
    void crossCheckMatching(cv::Mat& descriptors1, cv::Mat& descriptors2, std::vector<cv::DMatch>& filteredMatches12);
    void randomSearch();

public:
    Alien(const cv::FileNode& file);
    void save(const std::string filename);
    void init(const cv::Mat& frame1,const cv::Rect &box);
    float processFrame(const cv::Mat& img1,const cv::Mat& img2,cv::RotatedRect& bbnext,bool& lastboxfound);
};

#endif /* defined(__ALIEN__Alien__) */








