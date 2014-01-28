//
//  utils.cpp
//  ALIEN
//
//  Author: alantrrs
//  Modified by Edgar Reyna on 11/15/13.
//  Copyright (c) 2013 Edgar Reyna. All rights reserved.
//

#include "utils.h"

using namespace cv;
using namespace std;

// Tools needes by Alien

int factorial(int x) {
    if (x==0 || x==1) return 1;
    else return x * factorial(x - 1);
}

int nCr(int n, int r){
    int C=1;
    for (int i=0; i<r; i++) {
        C *= n;
        n--;
    }
    C = C/(factorial(r));
    return C;
}

double median(vector<double> m){
    int n = floor(m.size() / 2);
    nth_element(m.begin(), m.begin()+n, m.end());
    return m[n];
}

float median(vector<float> m){
    int n = floor(m.size() / 2);
    nth_element(m.begin(), m.begin()+n, m.end());
    return m[n];
}

float average(vector<float> av){
    float sum=0;
    for (vector<float>::iterator it=av.begin(); it!=av.end(); ++it)
        sum += *it;
    return sum/av.size();
}

vector<vector<int>> combinations(int size){
    vector<vector<int>> idx;
    vector<int> pt(2);
    if (size==2) {
        pt[0]=1, pt[1]=0;
        idx.push_back(pt);
    }else if (size>2){
        for (int i=size-2; i>=0; i--) {
            pt[0] = size-1, pt[1]=i;
            idx.push_back(pt);
        }
        vector<vector<int>> idx2 = combinations(size-1);
        idx.insert(idx.end(), idx2.begin(), idx2.end());
    }
    return idx;
}

Point2f getULPoint(RotatedRect rbox){
    Point2f ulPoint;
    Size2f bbSize = rbox.size;
    float d = sqrt((bbSize.width*bbSize.width)+(bbSize.height*bbSize.height))/2;
    float theta = rbox.angle + atan2(bbSize.height, bbSize.width);
    ulPoint.x = rbox.center.x - d*sin(theta);
    ulPoint.y = rbox.center.y - d*cos(theta);
    return ulPoint;
}

Mat createM(float x_, float y_, float rot){
    Mat M = Mat::zeros(3, 3, CV_32F);
    M.at<float>(0,0) = cos(rot);
    M.at<float>(0,1) = -sin(rot);
    M.at<float>(1,0) = sin(rot);
    M.at<float>(1,1) = cos(rot);
    M.at<float>(0,2) = x_;
    M.at<float>(1,2) = y_;
    M.at<float>(2,2) = 1;
    return M;
}

vector<DMatch> goodMatches(const vector<vector<DMatch>>& kmatches, float threshold){
    vector<DMatch> good_matches;
    good_matches.reserve(kmatches.size());
    for (size_t i = 0; i < kmatches.size(); ++i){
        if (kmatches[i].size() < 2) continue;
        const DMatch &m1 = kmatches[i][0];
        const DMatch &m2 = kmatches[i][1];
        if(m1.distance <= threshold * m2.distance) good_matches.push_back(m1);
    }
    return good_matches;
}

vector<int> index_shuffle(int begin,int end){
    vector<int> indexes(end-begin);
    for (int i=begin;i<end;i++){indexes[i]=i;}
    random_shuffle(indexes.begin(),indexes.end());
    return indexes;
}

void drawBox(Mat& image, cv::Rect box, Scalar color, int thick){
    rectangle( image, cvPoint(box.x, box.y), cvPoint(box.x+box.width,box.y+box.height),color, thick);
}

void drawBox(Mat& image, RotatedRect box, Scalar color, int thick){
    Point2f points[4];
    box.points(points);
    line( image, points[0] , points[1], color, thick ); //TOP line
    line( image, points[1] , points[2], color, thick );
    line( image, points[2] , points[3], color, thick );
    line( image, points[3] , points[0], color, thick );
}

// Draw object features as red circules, context features as blue crosses,
// excluded features as green circules and new features not classified as yellow crosses
void displayKeypoints(Mat& display, const vector<KeyPoint>& kpts){
    Scalar color;
    Point2f offx = Point2f(3,0);
    Point2f offy = Point2f(0,3);
    for (size_t i=0;i<kpts.size();i++){
        if (kpts[i].class_id == 1){
            color = Scalar(0,0,255);        //RED
            cv::circle(display,kpts[i].pt,3,color);
        }else if (kpts[i].class_id == 2){
            color = Scalar(0,255,0);        //GREEN
            cv::circle(display,kpts[i].pt,3,color);
        }else if (kpts[i].class_id == 0){
            color = Scalar(255,0,0);        //BLUE
            cv::line(display,kpts[i].pt-offx,kpts[i].pt+offx,color);
            cv::line(display,kpts[i].pt-offy,kpts[i].pt+offy,color);
        }else{
            color = Scalar(0,255,255);      //YELLOW
            cv::line(display,kpts[i].pt-offx,kpts[i].pt+offx,color);
            cv::line(display,kpts[i].pt-offy,kpts[i].pt+offy,color);
        }
        
    }
}

// Returns true if some point is inside the box
bool is_inside(const Point pt,const Rect& box){
    int x = pt.x;
    int y = pt.y;
    return (x>box.x && x<box.br().x &&  y>box.y && y<box.br().y);
}
bool evLine(const Point2f pt,Point2f pt0,Point2f pt1,Point2f pt2,Point2f pt3){
    float x = pt.x;
    float y = pt.y;
    Point2f diff;
    bool l1,l2,l3,l4;
    diff = pt1-pt0;
    l1 = (diff.y/diff.x)*(x-pt1.x)+pt1.y<y;
    diff = pt2-pt1;
    l2 = (diff.y/diff.x)*(x-pt2.x)+pt2.y<y;
    diff = pt3-pt2;
    l3 = (diff.y/diff.x)*(x-pt3.x)+pt3.y>y;
    diff = pt0-pt3;
    l4 = (diff.y/diff.x)*(x-pt0.x)+pt0.y>y;
    if (l1 && l2 && l3 && l4) return true;
    else return false;
}

bool is_inside(const Point2f pt,const RotatedRect& box){
    Point2f pts[4];
    box.points(pts);
    if (box.angle>=0 && box.angle<90) {
        return evLine(pt, pts[0], pts[1], pts[2], pts[3]);
    }else if (box.angle>=90 && box.angle<180){
        return evLine(pt, pts[3], pts[0], pts[1], pts[2]);
    }else if (box.angle>=180 && box.angle<270){
        return evLine(pt, pts[2], pts[3], pts[0], pts[1]);
    }else {
        //if (box.angle>=270 && box.angle<360)
        return evLine(pt, pts[1], pts[2], pts[3], pts[0]);
    }
}





