#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>

#include<string>

using namespace cv;
using namespace std;

void track_features(Mat img0, Mat img1) {
    //image 0
    vector<Point2f> img0_features;
    vector<KeyPoint> img0_kps;
    int max_corners = 150;
    double quality = 0.01, min_distance = 30;
    goodFeaturesToTrack(img0,img0_features,max_corners,quality,min_distance);
    for (int i=0; i<img0_features.size();i++) {
        Point2f pt_to_push0 = img0_features[i];
        KeyPoint img0_kp;
        img0_kp.pt = pt_to_push0;
        img0_kps.push_back(img0_kp);
        // cout<<"x:"<<img0_features[i].x<<", y:"<<img0_features[i].y<<endl;
        // cout<<"x:"<<kp.pt.x<<", y:"<<kp.pt.y<<endl;
    }

    //image 1
    vector<Point2f> img1_features;
    vector<KeyPoint> img1_kps;
    goodFeaturesToTrack(img1,img1_features,max_corners,quality,min_distance);
    for (int i=0; i<img1_features.size();i++) {
        Point2f pt_to_push1 = img1_features[i];
        KeyPoint img1_kp;
        img1_kp.pt = pt_to_push1;
        img1_kps.push_back(img1_kp);
        // cout<<"x:"<<img0_features[i].x<<", y:"<<img0_features[i].y<<endl;
        // cout<<"x:"<<kp.pt.x<<", y:"<<kp.pt.y<<endl;
    }

    //run BRIEF descriptor on Image 0
    cv::Mat img0_desc;
    Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief_img0 =  cv::xfeatures2d::BriefDescriptorExtractor::create(64); //i don't know what 64 means - it was in the example
    brief_img0->compute(img0, img0_kps, img0_desc); //feature_points_depth.points might need to be of type vector<KeyPoint>

    //run BRIEF descriptor on Image 1
    cv::Mat img1_desc;
    Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief_img1 =  cv::xfeatures2d::BriefDescriptorExtractor::create(64); //i don't know what 64 means - it was in the example
    brief_img1->compute(img1, img1_kps, img1_desc); //feature_points_depth.points might need to be of type vector<KeyPoint>

    // match the points
    vector<DMatch> matches;
    Ptr<BFMatcher> desc_matcher = cv::BFMatcher::create(cv::NORM_L2, true); //choose NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. 
    desc_matcher->match(img0_desc, img1_desc, matches, Mat());

    Ptr<DescriptorMatcher> flann_matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > flann_matches;
    flann_matcher->knnMatch(img0_desc, img1_desc, flann_matches, 2 );



    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < flann_matches.size(); i++)
    {
        if (flann_matches[i][0].distance < ratio_thresh * flann_matches[i][1].distance)
        {
            good_matches.push_back(flann_matches[i][0]);
        }
    }
    //-- Draw matches
    Mat img_matches;
    drawMatches(img0, img0_kps, img1, img1_kps, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    imshow("Good Matches", img_matches );
    waitKey();
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