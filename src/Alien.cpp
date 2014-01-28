//
//  Alien.cpp
//  ALIEN
//
//  Created by Edgar Reyna on 11/7/13.
//  Copyright (c) 2013 Edgar Reyna. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include "Alien.h"


using namespace std;
using namespace cv;


Alien::Alien(const FileNode& file){
    theRNG().state =1234;
    //Read parameters from file
    params_ = file;
    read(file);
    // Set features extractor
    featuresExtractor = SIFT(1000);
    // Set features matcher
    matcher_ = DescriptorMatcher::create("BruteForce");
}

void Alien::read(const FileNode& file){
    ///Bounding Box Parameters
    min_win = (int)file["min_win"];
    patch_size = (int)file["patch_size"];
    printf("Min_win: %d, patch: %d\n",min_win,patch_size);
}

void Alien::save(string filename){
    cv::FileStorage fs(filename,cv::FileStorage::WRITE);
    fs << "box " << init_box_;
    object.save(fs);
    context.save(fs);
    fs.release();
}

void Alien::init(const Mat& frame1,const Rect &box){
    init_box_ = box;
    lastvalid = true;
    Point2f center((box.x+box.br().x)/2, (box.y+box.br().y)/2);
    RotatedRect rbox(center,box.size(),0);
    frameSize.x = frame1.cols, frameSize.y = frame1.rows;
    lastbox = rbox;
    lastArea = searchArea(box);
    
    // Features extraction
    vector<KeyPoint> keypts;
    Mat descriptors;
    Mat mask = Mat::zeros(frame1.rows,frame1.cols,CV_8U);
    mask(lastArea).setTo(255);
    featuresExtractor(frame1,mask,keypts,descriptors);
    
    // Fill object keypoints
    vector<Point2f> points_;
    points_.reserve(keypts.size());
    Mat obj_descriptors = Mat(descriptors.rows,descriptors.cols,descriptors.type());
    Mat con_descriptors = Mat(descriptors.rows,descriptors.cols,descriptors.type());
    
    int j=0,k=0;
    for (int i=0;i<keypts.size();i++){
        if (is_inside(keypts[i].pt,box)){
            keypts[i].class_id = 1;
            points_.push_back(keypts[i].pt);
            descriptors.row(i).copyTo(obj_descriptors.row(k++));
        }else{
            keypts[i].class_id = 0;
            descriptors.row(i).copyTo(con_descriptors.row(j++));
        }
    }
    obj_descriptors.resize(k);
    con_descriptors.resize(j);
    
    // Display features
    Mat features;
    cvtColor(frame1, features, CV_GRAY2BGR);
    displayKeypoints(features, keypts);
    drawBox(features, box);
    drawBox(features, lastArea);
    imshow("Features", features);
    printf("Keypoints size: %zu, Inside box: %zu\n",keypts.size(),points_.size());
    
    // init model
    // change keypoint coordinates
    vector<Point2f> pts(points_.size());
    for (int i=0; i<points_.size(); i++) {
        pts[i].x = points_[i].x - box.x;
        pts[i].y = points_[i].y - box.y;
    }
    object.init(frame1(box), obj_descriptors, pts, rbox);
    lastM = createM(box.x, box.y, 0);
    
    // init context model
    context.init(con_descriptors);
}

float Alien::processFrame(const Mat& img1, const Mat& img2, RotatedRect& bbnext, bool& lastboxfound){
    // Features extraction
    vector<KeyPoint> keypts;
    Mat descriptors;
    vector<vector<DMatch>> obj_mat_k, con_mat_k;
    Mat mask = Mat::zeros(img2.rows,img2.cols,CV_8U);
    mask(lastArea).setTo(255);
    featuresExtractor(img2,mask,keypts,descriptors);
    printf("Keypoints size: %zu\n",keypts.size());
    
    // Matching
    Mat obj_descriptors = object.getDescriptors();
    Mat con_descriptors = context.getDescriptors();
    vector<DMatch> obj_matches, con_matches;
    crossCheckMatching(descriptors, obj_descriptors, obj_matches);
    crossCheckMatching(descriptors, con_descriptors, con_matches);
    printf("obj_matches size: %zu, con_matches size: %zu\n",obj_matches.size(),con_matches.size());
    for (int i=0; i<con_matches.size(); i++)
        keypts[con_matches[i].queryIdx].class_id = 0;
    for (int i=0; i<obj_matches.size(); i++)
        keypts[obj_matches[i].queryIdx].class_id = 1;

    // Remove features that match both sets F_t = T_t*\C_t*
    Mat obj_descriptors_ = removeFeatures(descriptors, keypts, obj_matches, con_matches);
    
    // Get transformation using MLESAC on F_t
    Mat temp_descriptors;
    vector<Point2f> obj_pts, temp_pts;
    vector<DMatch> model_matches;
    obj_pts.reserve(keypts.size());
    for (int i=0; i<obj_matches.size(); i++)
        obj_pts.push_back(keypts[obj_matches[i].queryIdx].pt);
    object.getTemplate(temp_descriptors, temp_pts);
    crossCheckMatching(obj_descriptors_, temp_descriptors, model_matches);
    
    Mat M = homography(obj_pts, temp_pts, model_matches);
    //float errorMLESAC = mlesac(obj_pts, temp_pts, model_matches, M);
    
    // if (displacement< max_displacement && rotation < max_rotation) -> obj_detected
    RotatedRect dbb = lastbox;
    if (validM) {
        lastvalid = checkM(M);
        newBox(M, dbb);
    }
    if (lastvalid && validM) {
    // E_t = All features inside new bounding box are labeled as object features
        obj_pts.clear();
        obj_descriptors_.resize(descriptors.rows);
        Mat occ_descriptors(descriptors.rows,descriptors.cols, descriptors.type());
        Mat con_descriptors_(descriptors.rows,descriptors.cols, descriptors.type());
        int j=0, k=0, l=0;
        for (int i=0; i<keypts.size(); i++) {
            KeyPoint kpt= keypts[i];
            if (is_inside(kpt.pt, dbb)) {
                descriptors.row(i).copyTo(obj_descriptors_.row(j++));
                obj_pts.push_back(kpt.pt);
                // O_t = All features on C_t inside the box are labeled as occlusion features
                if (kpt.class_id==0)
                    descriptors.row(i).copyTo(occ_descriptors.row(k++));
            }else{
                descriptors.row(i).copyTo(con_descriptors_.row(l++));
            }
        }
        obj_pts.resize(j);
        obj_descriptors_.resize(j);
        occ_descriptors.resize(k);
        con_descriptors_.resize(l);
        
        framesLost = 0;
        bbnext = dbb;
        lastbox = dbb;
        lastM = M;
        lastArea = searchArea(dbb.boundingRect());

        // if (O_t<max_occlussion_features) --> update_models
        if (occ_descriptors.rows < obj_descriptors_.rows*0.5) {
            printf("Updating models\n");
            // Update Object Features
            // E_t' = Et are transformed to the template's coordinate frame
            temp_pts.clear();
            Mat o_pts((int)obj_pts.size(),3,CV_32F);
            for (int j=0; j<obj_pts.size(); j++) {
                o_pts.at<float>(j,0) = obj_pts[j].x;
                o_pts.at<float>(j,1) = obj_pts[j].y;
                o_pts.at<float>(j,2) = 1;
            }
            Mat invM = M.inv();
            Mat t_pts = o_pts*invM.t();
            temp_pts.resize(t_pts.rows);
            for (int j=0; j<t_pts.rows; j++) {
                temp_pts[j].x = t_pts.at<float>(j,0);
                temp_pts[j].y = t_pts.at<float>(j,1);
            }
            
            // T_t = Add E_t' features to T_t-1
            object.update(matcher_, obj_descriptors_, temp_pts);
            // Update context features:
            context.update(matcher_, con_descriptors_, occ_descriptors);
            printf("Keypoints size: %zu, Inside box: %zu\n",keypts.size(),obj_pts.size());
            
            //TODO: Forget features
            //      if |T_t| > N_T then keep only N_T random features from T_t
            //      if |D_t| > N_D then keep only N_D random features from D_t
        }
    }else {
        framesLost++;
        if (framesLost >= 10) {
            // TODO: init random search if frameLost>10
            randomSearch();
        }
    }
    
    Mat features;
    cvtColor(img2, features, CV_GRAY2BGR);
    displayKeypoints(features, keypts);
    drawBox(features, dbb);
    drawBox(features, lastArea);
    imshow("Features", features);
    lastboxfound=lastvalid;
    return 0;
}


// Remove features that match both sets F_t = T_t*\C_t*
// Return descriptors of remaining features
Mat Alien::removeFeatures(const Mat& descriptors, vector<KeyPoint>& keypts, vector<DMatch>& obj_matches, vector<DMatch>& con_matches){
    vector<vector<DMatch>> rmatches;
    vector<DMatch> remove_matches,obj_matches_aux, con_matches_aux;
    Mat obj_descrip((int)obj_matches.size(),descriptors.cols,descriptors.type());
    Mat con_descrip((int)con_matches.size(),descriptors.cols,descriptors.type());;
    for (int i=0; i<obj_matches.size(); i++) {descriptors.row(obj_matches[i].queryIdx).copyTo(obj_descrip.row(i));}
    for (int i=0; i<con_matches.size(); i++) {descriptors.row(con_matches[i].queryIdx).copyTo(con_descrip.row(i));}
    crossCheckMatching(obj_descrip, con_descrip, remove_matches);
    if (remove_matches.size()>1) {
        printf("Removed features: %zu\n", remove_matches.size());
        vector<int> obj_idx;
        obj_idx.push_back(-1);
        for (int i=0; i<remove_matches.size(); i++) {
            obj_idx.push_back(remove_matches[i].queryIdx);
        }
        //TODO: Is obj_idx already sorted?
        sort(obj_idx.begin(), obj_idx.end());
        for (int i=0; i<obj_idx.size()-1; i++) {
            obj_matches_aux.insert(obj_matches_aux.end(),obj_matches.begin()+obj_idx[i]+1, obj_matches.begin()+obj_idx[i+1]);}
        obj_matches.resize(obj_matches_aux.size());
        obj_matches.assign(obj_matches_aux.begin(), obj_matches_aux.end());
    }else if (remove_matches.size()==1){
        printf("One feature to remove\n");
        int idx = remove_matches[0].queryIdx;
        obj_matches_aux.assign(obj_matches.begin()+idx+1, obj_matches.end());
        obj_matches.resize(idx);
        obj_matches.insert(obj_matches.end(), obj_matches_aux.begin(), obj_matches_aux.end());
    }else{
        printf("No features to remove\n");
        return obj_descrip;
    }
    for (int i=0; i<obj_matches.size(); i++)
        descriptors.row(obj_matches[i].queryIdx).copyTo(obj_descrip.row(i));
    obj_descrip.resize(obj_matches.size());
    for (int i=0; i<remove_matches.size(); i++)
        keypts[remove_matches[i].queryIdx].class_id = 2;
    
    return obj_descrip;
}


Mat Alien::homography(const vector<Point2f>& queryPts, const vector<Point2f>& modelPts, vector<DMatch>& matches){
    Mat hM;
    validM = true;
    vector<Point2f> srcPts, dstPts;
    for (int i=0; i<matches.size(); i++) {
        srcPts.push_back(modelPts[matches[i].trainIdx]);
        dstPts.push_back(queryPts[matches[i].queryIdx]);
    }
    try {
        hM = findHomography(srcPts, dstPts, RANSAC);
    } catch (exception& e) {
        cerr << e.what();
        validM = false;
    }
    Mat M(3,3,CV_32F);
    if (validM) {
        Point2f origin = Point2f(hM.at<double>(0,2), hM.at<double>(1,2));
        vector<float> rot;
        rot.reserve(dstPts.size());
        for (int j=0; j< dstPts.size(); j++){
            Point2f newPt = dstPts[j]-origin;
            rot.push_back(atan2(newPt.y, newPt.x)-atan2(srcPts[j].y, srcPts[j].x));
        }
        float rotation = median(rot);
        M = createM(origin.x, origin.y, rotation);
        printf("New origin: %f, %f, Rotation: %f\n", origin.x, origin.y, rotation);
    }
    return M;
}


// Symmmetric transfer error  (eq 12 in ALIEN paper)
float symTransError(const Point2f qPt, const Point2f mPt, const Point2f tqPt, const Point2f tmPt){
    float trasnErr1 = sqrt((qPt.x-tmPt.x)*(qPt.x-tmPt.x)+(qPt.y-tmPt.y)*(qPt.y-tmPt.y));
    float trasnErr2 = sqrt((mPt.x-tqPt.x)*(mPt.x-tqPt.x)+(mPt.y-tqPt.y)*(mPt.y-tqPt.y));
    return trasnErr1+trasnErr2;
}

float Alien::mlesac(const vector<Point2f>& queryPts, const vector<Point2f>& modelPts, vector<DMatch>& matches, Mat& best_M){
    // Set parameters
    int min_matches = 2;
    float threshold = 5.0; //5 pixels
    float min_cost = FLT_MAX;
    float min_rot = FLT_MAX;
    float avError = 0;
    float gamma = 0.5; //TODO: Get correct params for gamma,sigma,v
    float sigma = 5.0;
    float v = 0.5;
    int max_iterations = nCr((int)matches.size(), min_matches);
    if (max_iterations>200) max_iterations=200;
    vector<vector<int>> idx_matches = combinations((int)matches.size());
    random_shuffle(idx_matches.begin(), idx_matches.end());
    printf("idx size: %zu\n", idx_matches.size());
    // Start iteration
    for (int i=0;i<max_iterations;i++){
        // Randomly select the smallest subset of data
        vector<Point2f> i_qPts, i_mPts;
        for (int j=0; j< min_matches; j++){
            DMatch m = matches[idx_matches[i][j]];
            i_mPts.push_back(modelPts[m.trainIdx]);
            i_qPts.push_back(queryPts[m.queryIdx]);
        }
        // model parameters
        float alpha = atan2(i_mPts[1].y, i_mPts[1].x) - atan2(i_mPts[0].y, i_mPts[0].x);
        float beta = asin(norm(i_mPts[0])*sin(alpha)/norm(i_mPts[1] - i_mPts[0]));
        // linear equations
        Point2f diff = i_qPts[1] - i_qPts[0];
        float x_ = i_qPts[0].x - ((((diff.y/diff.x)+tan(beta))*diff.x - diff.y)/tan(alpha));
        float y_ = i_qPts[1].y - ((((diff.y/diff.x)+tan(beta)+tan(alpha))*((diff.y/diff.x)+tan(beta))*diff.x + ((diff.y/diff.x)+tan(beta))*(-diff.y))/tan(alpha));
        // traslation and rotation
        Point2f origen(x_,y_);
        vector<Point2f> newPts;
        newPts.reserve(i_qPts.size());
        for (int j=0; j< i_qPts.size(); j++){
            newPts.push_back(i_qPts[j]-origen);
        }
        float rot = atan2(newPts[0].y, newPts[0].x)-atan2(i_mPts[0].y, i_mPts[0].x);
        float rot2 = atan2(newPts[1].y, newPts[1].x)-atan2(i_mPts[1].y, i_mPts[1].x);
        if (abs(rot-rot2) > 0.2) continue;      // Not a correct match
        // Initialize model with data
        Mat M = createM(x_, y_, rot);
        // Expand hypothetical inliers with all points inside a threshold
        Mat query((int)queryPts.size(),3,CV_32F);
        Mat model((int)modelPts.size(),3,CV_32F);
        for (int j=0; j<queryPts.size(); j++) {
            query.at<float>(j,0) = queryPts[j].x;
            query.at<float>(j,1) = queryPts[j].y;
            query.at<float>(j,2) = 1;
        }
        for (int j=0; j<modelPts.size(); j++) {
            model.at<float>(j,0) = modelPts[j].x;
            model.at<float>(j,1) = modelPts[j].y;
            model.at<float>(j,2) = 1;
        }
        
        Mat invM = M.inv();
        Mat tqPts = query*invM.t();
        Mat tmPts = model*M.t();
        vector<Point2f> t_qPts(tqPts.rows);
        vector<Point2f> t_mPts(tmPts.rows);
        for (int j=0; j<t_qPts.size(); j++) {
            t_qPts[j].x = tqPts.at<float>(j,0);
            t_qPts[j].y = tqPts.at<float>(j,1);
        }
        for (int j=0; j<t_mPts.size(); j++) {
            t_mPts[j].x = tmPts.at<float>(j,0);
            t_mPts[j].y = tmPts.at<float>(j,1);
        }
        float Err = 0;
        float current_cost = 0;
        i_mPts.clear(), i_qPts.clear();
        i_mPts.reserve(matches.size());
        i_qPts.reserve(matches.size());
        for (int j=0; j<matches.size(); j++){
            // Symmmetric transfer error
            float error = symTransError(queryPts[matches[j].queryIdx], modelPts[matches[j].trainIdx], t_qPts[matches[j].queryIdx], t_mPts[matches[j].trainIdx]);
            Err += error;
            current_cost += -log((gamma*exp(-pow(error,(float)2.0)/(2*pow(sigma,(float)2.0))))/sqrt(2*PI*sigma)+(1-gamma)/v);
            if (error < threshold){     // points supporting the model
                i_qPts.push_back(queryPts[matches[j].queryIdx]);
                i_mPts.push_back(modelPts[matches[j].trainIdx]);
            }
        }
        // Repeat and keep model with lowest cost
        if (current_cost < min_cost){
                min_cost = current_cost;
                min_rot = abs(rot);
                avError = Err;
                M.copyTo(best_M);
        }
    }
    printf("Cost: %f\n", min_cost);
    cout << best_M << endl;
    return avError;
}


// if (displacement< max_displacement && rotation < max_rotation) -> obj_detected
bool Alien::checkM(Mat& M){
    Point2f lastOrigin(lastM.at<float>(0,2), lastM.at<float>(1,2));
    Point2f newOrigin(M.at<float>(0,2),M.at<float>(1,2));
    Point2f d(newOrigin.x - lastOrigin.x, newOrigin.y - lastOrigin.y);
    float displacement = norm(d);
    float rotation = atan2(M.at<float>(1,0), M.at<float>(0,0))-atan2(lastM.at<float>(1,0), lastM.at<float>(0,0));
    printf("Displacement: %f Rotation: %f\n", displacement,rotation);
    if (displacement<15 && abs(rotation)<0.2){
        printf("Object Detected\n");
        lastDisp = d;
        lastRot = rotation;
        return true;
    }else{
        printf("Fast movement, Object not detected\n");
        Point2f newOrigin = lastOrigin + lastDisp;
        float rotation = atan2(lastM.at<float>(1,0), lastM.at<float>(0,0)) + lastRot;
        M = createM(newOrigin.x, newOrigin.y, rotation);
        return false;
    }
}


// Create new rBox with the transformation matrix
void Alien::newBox(const Mat& M, RotatedRect& rbox){
    Point2f lastOrigin(lastM.at<float>(0,2), lastM.at<float>(1,2));
    Point2f newOrigin(M.at<float>(0,2),M.at<float>(1,2));
    Point2f template_c = rbox.center - lastOrigin;
    Mat temp_c(3,1,CV_32F);
    temp_c.at<float>(0) = template_c.x;
    temp_c.at<float>(1) = template_c.y;
    temp_c.at<float>(2) = 1;
    Mat n_center = M*temp_c;
    Point2f newCenter(n_center.at<float>(0), n_center.at<float>(1));
    rbox.center = newCenter;
    if (lastRot != 0) {
        Point2f ulPoint = getULPoint(rbox);
        Point2f diff1 = newCenter - ulPoint;
        Point2f diff2 = newCenter - newOrigin;
        float rho = atan2(diff2.y,diff2.x) - atan2(diff1.y,diff1.x);
        rbox.angle += rho;
        if (rbox.angle<0) rbox.angle = 2*PI+rbox.angle;
        if (rbox.angle>=2*PI) rbox.angle = rbox.angle-2*PI;
    }
}


void Alien::crossCheckMatching(Mat& descriptors1, Mat& descriptors2, vector<DMatch>& filteredMatches12){
    filteredMatches12.clear();
    vector<DMatch> matches12, matches21;
    matcher_ -> match(descriptors1, descriptors2, matches12);
    matcher_ -> match(descriptors2, descriptors1, matches21);
    for(int i = 0; i<matches12.size(); i++ ){
        DMatch forward = matches12[i];
        DMatch backward = matches21[forward.trainIdx];
        if (backward.trainIdx == forward.queryIdx){
            filteredMatches12.push_back(forward);
        }
    }
}


Rect Alien::searchArea(const Rect &box){
    Point2f center((box.x+box.br().x)/2,(box.y+box.br().y)/2);
    Rect area;
    int height = box.height*2.5;
    int width = box.width*2.5;
    if (center.x>=width) {
        area.x = center.x - width;
        area.width = width;
    }else {
        area.x = 0;
        area.width = center.x;}
    if (center.y>=height){
        area.y = center.y - height;
        area.height = height;
    }else {
        area.y = 0;
        area.height = center.y;}
    if (center.x+width<frameSize.x) {
        area.width += width;
    }else {
        area.width += frameSize.x-center.x;}
    if (center.y+height<frameSize.y) {
        area.height += height;
    }else {
        area.height += frameSize.y-center.y;}
    return area;
}


void Alien::randomSearch(){
    // RandomSearch applied when the object hasn´t been detected for certain frames
    
    
    
    
}
