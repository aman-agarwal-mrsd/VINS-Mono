#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include<string>

using namespace cv;
using namespace std;

void track_features(Mat img0, Mat img1) {
    vector<Point2f> img0_features;
    vector<KeyPoint> kps;
    int max_corners = 150;
    double quality = 0.01, min_distance = 30;
    goodFeaturesToTrack(img0,img0_features,max_corners,quality,min_distance);
    for (int i=0; i<img0_features.size();i++) {
        Point2f pt(img0_features[i].x,img0_features[i].y);
        KeyPoint kp(pt);
        kps.push_back(kp);
        // cout<<"x:"<<img0_features[i].x<<", y:"<<img0_features[i].y<<endl;
    }
}

int main()
{
    string img0_file_name = "left.jpg", img1_file_name = "right.jpg";
    Mat img0, img1;
    img0 = imread(img0_file_name, CV_LOAD_IMAGE_COLOR);   // Read the file
    img1 = imread(img1_file_name, CV_LOAD_IMAGE_COLOR);   // Read the file

    Mat img0_gray, img1_gray;
    cvtColor(img0, img0_gray, CV_BGR2GRAY );
    cvtColor(img1, img1_gray, CV_BGR2GRAY );

    track_features(img0_gray,img1_gray);

    // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    // imshow( "Display window", img1);                   // Show our image inside it.

    // waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}