//
//  Context.cpp
//  ALIEN
//
//  Created by Edgar Reyna on 12/6/13.
//  Copyright (c) 2013 Edgar Reyna. All rights reserved.
//
#include <opencv2/opencv.hpp>
#include "Context.h"

using namespace std;
using namespace cv;

void Context::save( FileStorage& file){
    file << "Context Features" << "[";
    file << descriptors;
    file << "]";
    file << "Occlusion Features" << "[";
    file << occ_descriptors;
    file << "]";
}

void Context::init(Mat& descriptors_){
    descriptors = descriptors_;
    time = vector<int>(descriptors.rows,1);
    oversample = vector<int>(descriptors.rows,1);
    occ_descriptors.push_back(descriptors_.row(0));
}

// descriptors_ -> features from the anular area of the OBB
// occ_ -> occlusion features inside the OBB
void Context::update(Ptr<DescriptorMatcher> matcher, Mat& descriptors_, Mat& occ_){
    vector<DMatch> matches;
    // D_t = Add O_t to D_t-1
    matcher -> match(occ_,occ_descriptors,matches);
    for (int i=0; i<matches.size(); i++) {
        if (matches[i].distance>100) { //50
            occ_descriptors.push_back(occ_.row(matches[i].queryIdx));
        }
    }
    matches.clear();
    // add C_t-1 to D_t
    Mat update_descriptors;
    if (descriptors_.rows>occ_descriptors.rows) {
        matcher -> match(occ_descriptors,descriptors_,matches);
        descriptors_.copyTo(update_descriptors);
        for (int i=0; i<matches.size(); i++) {
            if (matches[i].distance>100) {
                update_descriptors.push_back(occ_descriptors.row(matches[i].queryIdx));
            }
        }
    }else{
        matcher -> match(descriptors_,occ_descriptors,matches);
        occ_descriptors.copyTo(update_descriptors);
        for (int i=0; i<matches.size(); i++) {
            if (matches[i].distance>100) {
                update_descriptors.push_back(descriptors_.row(matches[i].queryIdx));
            }
        }
    }
    // C_t = Add all context features accumulated during a l temporal window to D_t
    matches.clear();
    matcher -> match(update_descriptors, descriptors, matches);
    for (int i=0; i<matches.size(); i++) {
        if (matches[i].distance<100) {
            oversample[matches[i].trainIdx]++;
        }else{
            descriptors.push_back(update_descriptors.row(matches[i].queryIdx));
            time.push_back(1);
            oversample.push_back(1);
        }
    }
    timeWindow();
}

void eliminate(vector<int>& source, int idx){
    vector<int> aux;
    aux.assign(source.begin()+idx+1, source.end());
    source.resize(idx);
    source.insert(source.end(), aux.begin(), aux.end());
}

void eliminate(Mat& source, int idx){
    for (int i=idx+1; i<source.rows; i++) {
        source.row(i).copyTo(source.row(i-1));}
    source.pop_back();
}

void Context::timeWindow(){
    for (int i=0; i<descriptors.rows; i++) {
        if (time[i]>=9){
            if (oversample[i]>1) {
                oversample[i]--;
            }else{
                eliminate(time, i);
                eliminate(oversample, i);
                eliminate(descriptors, i);
            }
        }else
            time[i]++;
    }
}