//
//  Object.cpp
//  ALIEN
//
//  Created by Edgar Reyna on 1/17/14.
//  Copyright (c) 2014 Edgar Reyna. All rights reserved.
//
#include <opencv2/opencv.hpp>
#include "Object.h"

using namespace std;
using namespace cv;

void Object::save( FileStorage& file){
    file << "Object Features" <<"[";
    file << descriptors;
    file << "]";
}

void Object::init(const Mat& img, Mat& descriptors_, vector<Point2f>& keypts, RotatedRect rbox){
    // Object parameters
    descriptors = descriptors_;
    init_patch = img;
    
    // Init object template
    obj.bb = rbox;
    obj.patch = img;
    obj.keypoints.assign(keypts.begin(), keypts.end());
    obj.descriptors = descriptors_;
}

void Object::update(Ptr<DescriptorMatcher>& matcher, Mat& descriptors_, vector<Point2f>& keypts){
    obj.keypoints.assign(keypts.begin(), keypts.end());
    descriptors_.copyTo(obj.descriptors);
    vector<DMatch> update_matches;
    matcher -> match(descriptors_, descriptors, update_matches);
    for (int i=0; i<update_matches.size(); i++) {
        if (update_matches[i].distance>50) {
            descriptors.push_back(descriptors_.row(update_matches[i].queryIdx));
        }
    }
}


void Object::getTemplate(Mat& descriptors_, vector<Point2f>& keypts){
    obj.descriptors.copyTo(descriptors_);
    keypts.assign(obj.keypoints.begin(), obj.keypoints.end());
}