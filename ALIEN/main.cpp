//
//  main.cpp
//  ALIEN
//
//  Author: alantrrs
//  Modified by Edgar Reyna on 11/15/13.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "Alien.h"
#include "utils.h"

using namespace cv;
using namespace std;

//Global variables
Rect box;
bool drawing_box = false;
bool gotBB = false;

const char* keys = {
    "{ i  |init_frame  |0     | initial frame }"
    "{ p  |parameters  |null  | Parameters file }"
    "{ b  |bb_init     |null  | initial box file }"
    "{ s  |source      |null  | video file }"
};

void readBB(const char* file);
void mouseHandler(int event, int x, int y, int flags, void *param);
void print_help(char** argv);


int main(int argc, char * argv[]){
    VideoCapture capture;
    FileStorage fs;
    FileStorage detector_file;
    bool fromfile=false;
    //Read options
    CommandLineParser parser(argc, argv, keys);
    int init_frame = parser.get<int>("i");
    string param_file = parser.get<string>("p");
    string video = parser.get<string>("s");
    string init_bb  = parser.get<string>("b");
    
    fs.open(param_file, FileStorage::READ);
    if (video != "null"){
        fromfile=true;
        capture.open(video);
    }else{
        fromfile=false;
        capture.open(0);
    }
    if (init_bb !="null"){
        readBB(init_bb.c_str());
        gotBB =true;
    }
    
    //Init camera
    if (!capture.isOpened()){
        cout << "capture device failed to open!" << endl;
        return 1;
    }
    //Register mouse callback to draw the bounding box
    cvNamedWindow("Tracker",CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Features",CV_WINDOW_AUTOSIZE);
    cvSetMouseCallback( "Tracker", mouseHandler, NULL );
    
    FILE  *bb_file = fopen("bounding_boxes.txt","w");
    
    Mat frame;
    Mat last_gray;
    Mat first;
    if (fromfile){
        capture.set(CV_CAP_PROP_POS_FRAMES,init_frame);
        capture.read(frame);
        last_gray.create(frame.rows,frame.cols,CV_8U);
        cvtColor(frame, last_gray, CV_BGR2GRAY);
        frame.copyTo(first);
    }

    ///Initialization
GETBOUNDINGBOX:
    while(!gotBB){
        if (!fromfile) capture.read(frame);
        else first.copyTo(frame);
        cvtColor(frame, last_gray, CV_BGR2GRAY);
        drawBox(frame,box);
        imshow("Tracker", frame);
        if (cvWaitKey(30) == 'q')
            return 0;
    }
    if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"]){
        cout << "Bounding box too small, try again." << endl;
        gotBB = false;
        goto GETBOUNDINGBOX;
    }
    drawBox(frame,box);
    imshow("Tracker", frame);
    //Remove callback
    cvSetMouseCallback( "Tracker", NULL, NULL );
    printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
    //Output file
    fprintf(bb_file,"%d,%d,%d,%d,%f\n",box.x,box.y,box.br().x,box.br().y,1.0);
    
INIT:
    // Framework
    Alien tracker(fs.getFirstTopLevelNode());
    tracker.init(last_gray,box);
    
    cvWaitKey();
    ///Run-time
    Mat current_gray;
    RotatedRect pbox;
    bool status=true;
    int frames = 1;
    int detections = 1;
    float conf;
    while(capture.read(frame)){
        cvtColor(frame, current_gray, CV_BGR2GRAY);
        cout << endl;
        //Process Frame
        double t=(double)getTickCount();
        conf = tracker.processFrame(last_gray,current_gray,pbox,status);
        t = ((double)getTickCount() - t)*1000/getTickFrequency();
        //Draw Box
        if (status){
            drawBox(frame,pbox);
            fprintf(bb_file,"%f,%f,%f,%f,%f,%f,%f\n",pbox.center.x, pbox.center.y, pbox.size.height,pbox.size.width, pbox.angle,conf,t);
            detections++;
        }
        else{
            fprintf(bb_file,"NaN,NaN,NaN,NaN,%f,%f\n",conf,t);
        }
        //Display
        imshow("Tracker", frame);
        swap(last_gray,current_gray);
        frames++;
        printf("Detection rate: %d/%d, period: %fms\n",detections,frames,t);
        if (cvWaitKey(30) == 'q') break;
    }
    tracker.save("Detector.yml");
    fclose(bb_file);
    capture.release();
    return 0;
}

void readBB(const char* file){
    ifstream bb_file (file);
    string line;
    getline(bb_file,line);
    istringstream linestream(line);
    string x1,y1,x2,y2;
    getline (linestream,x1, ',');
    getline (linestream,y1, ',');
    getline (linestream,x2, ',');
    getline (linestream,y2, ',');
    int x = atoi(x1.c_str());
    int y = atoi(y1.c_str());
    int w = atoi(x2.c_str())-x;
    int h = atoi(y2.c_str())-y;
    box = Rect(x,y,w,h);
}
//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
    switch( event ){
        case CV_EVENT_MOUSEMOVE:
            if (drawing_box){
                box.width = x-box.x;
                box.height = y-box.y;
            }
            break;
        case CV_EVENT_LBUTTONDOWN:
            drawing_box = true;
            box = Rect( x, y, 0, 0 );
            break;
        case CV_EVENT_LBUTTONUP:
            drawing_box = false;
            if( box.width < 0 ){
                box.x += box.width;
                box.width *= -1;
            }
            if( box.height < 0 ){
                box.y += box.height;
                box.height *= -1;
            }
            gotBB = true;
            break;
    }
}

void print_help(char** argv){
    printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
    printf("-s    source video\n");
    cout << "Options: " << endl;
    cout << "-b init_box_file.txt    init bounding box from file" << endl;
}
