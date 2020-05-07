#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

#include<string>

using namespace cv;
using namespace std;

void track_features(Mat img0, Mat img1) {

    //Account for distortion in intrinsics
    float fx = 4.6115862106007575e+02, fy = 4.5975286598073296e+02, cx = 3.6265929181685937e+02, cy = 2.4852105668448124e+02;
    Mat intrinsic_cam0 = (Mat1d(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    float k1 = -2.9545645106987750e-01, k2 = 8.6623215640186171e-02, p1 = 2.0132892276082517e-06, p2 = 1.3924531371276508e-05;
    Mat distortion_coefficients_cam0 = (Mat1d(1, 4) << k1, k2, p1, p2);
    // Mat img_size_cam0 = (Mat1d(1, 2) << 752, 480);
    double alpha = 1;
    Mat new_intrinsic_cam0 = getOptimalNewCameraMatrix(intrinsic_cam0,distortion_coefficients_cam0,{752,480},alpha);
    Mat new_intrinsic_cam1 = new_intrinsic_cam0; 

    //get transform from camera0 to camera 1
    cv::Mat body2cam0 = (cv::Mat1d(4,4) << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975, 0.999557249008, 0.0149672133247, 0.025715529948,  -0.064676986768, -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949, 0, 0, 0, 1);
    cv::Mat body2cam1 = (cv::Mat1d(4,4) << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556, 0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024, -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038, 0, 0, 0, 1);
    cv::Mat cam1_to_cam0_temp = body2cam1.inv() * body2cam0;
    cv::Mat cam1_to_cam0 = cam1_to_cam0_temp.rowRange(0,3);
    cv::Mat cam0_to_cam0 = (cv::Mat1d(3,4) <<   1.0, 0.0, 0.0, 0.0, 
                                                0.0, 1.0, 0.0, 0,0,
                                                0.0, 0.0, 1.0, 0.0 );
        // cv::Matx44f cam0_to_cam0(   1.0, 0.0, 0.0, 0.0, 
        //                         0.0, 1.0, 0.0, 0,0,
        //                         0.0, 0.0, 1.0, 0.0 );

    // Creating final projection matrixes for both cameras  
    cv::Mat cam0_proj = new_intrinsic_cam0 * cam0_to_cam0;
    cv::Mat cam1_proj = new_intrinsic_cam1 * cam1_to_cam0;
        
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
    // vector<DMatch> matches;
    // Ptr<BFMatcher> desc_matcher = cv::BFMatcher::create(cv::NORM_L2, true); //choose NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. 
    // desc_matcher->match(img0_desc, img1_desc, matches, Mat());

    if(img0_desc.type()!=CV_32F) {
        img0_desc.convertTo(img0_desc, CV_32F);
    }

    if(img1_desc.type()!=CV_32F) {
        img1_desc.convertTo(img1_desc, CV_32F);
    }

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

    //triangulation
    vector<cv::Point2d> triangulation_points0, triangulation_points1;

    for (size_t j = 0; j<good_matches.size(); j++)
    {
        triangulation_points0.push_back(img0_kps[good_matches[j].queryIdx].pt);
        triangulation_points1.push_back(img1_kps[good_matches[j].trainIdx].pt);
        //cout<<"dist: "<<good_matches[j].distance<<endl;
        //cout<<"img idx: "<<good_matches[j].imgIdx<<endl;
        //cout<<"query idx: "<<good_matches[j].queryIdx<<endl;
        //cout<<"img0 kp: "<<img0_kps[good_matches[j].queryIdx].pt<<endl;
        //cout<<"train idx: "<<good_matches[j].trainIdx<<endl;
        //cout<<"img1 kp: "<<img1_kps[good_matches[j].trainIdx].pt<<endl;
        //cout<<"triang0 "<<triangulation_points0[j]<<endl;
        //cout<<"triang1"<<triangulation_points1[j]<<endl;
        //cout<<"triang0 size"<<triangulation_points0.size()<<endl;
        //cout<<"triang1 size"<<triangulation_points1.size()<<endl;
    }

    // TODO Replace N by number of points
    cv::Mat pnts3D;// Output Matrix

    // // Create input points matrix
    // cv::Mat cam0pnts(1,N,CV_64FC2);
    // cv::Mat cam1pnts(1,N,CV_64FC2);
    // Populating pnts3D
    cv::triangulatePoints(cam0_proj,cam1_proj,triangulation_points0,triangulation_points1,pnts3D);
    

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