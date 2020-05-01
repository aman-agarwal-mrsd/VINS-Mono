#include <iostream>

#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

using namespace std;



sensor_msgs::PointCloud FeatureTracker::computeDepthMap(const sensor_msgs::ImageConstPtr &img_msg0, const sensor_msgs::ImageConstPtr &img_msg1, const sensor_msgs::PointCloudConstPtr &feature_points)
{

    ROS_INFO("Computing Depth Map");
    int numDisparities=16; // this must be a multiple of 16, number of depths to calc
    int blockSize=9; // this must be an off number, the smaller it is the more detailed the disparity map but higher likelihood for wrong correspondence
    
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(numDisparities,blockSize);

    cv::Mat disp; //make disparity map var
    ROS_INFO("Copy image 0");
    sensor_msgs::Image img0;
    img0.header = img_msg0->header;
    img0.height = img_msg0->height;
    img0.width = img_msg0->width;
    ROS_INFO("Left Image height: %d", img0.height);
    ROS_INFO("Left Image width: %d", img0.width);
    //img0.is_bigendian = img_msg0->is_bigendian; //I don't know what this is
    //img0.step = img_msg0->step;
    img0.data = img_msg0->data;
    //img0.encoding = "mono8"; //not sure if this is needed
    
    ROS_INFO("Copy image 1");
    sensor_msgs::Image img1;
    img1.header = img_msg1->header;
    img1.height = img_msg1->height;
    img1.width = img_msg1->width;
    ROS_INFO("Right Image height: %d", img1.height);
    ROS_INFO("Right Image width: %d", img1.width);
    //img0.is_bigendian = img_msg0->is_bigendian; //I don't know what this is
    //img1.step = img_msg1->step;
    img1.data = img_msg1->data;
    //img0.encoding = "mono8"; //not sure if this is needed
    
    //detect features in image 1
    vector<cv::Point2f> im1_pts;// I think this needs to be of type vector<KeyPoint>
    setMask();
    cv::goodFeaturesToTrack(forw_img, im1_pts, feature_points->points.size(), 0.01, MIN_DIST, mask);


    // //bm->setSpeckleRange(32);
    // ROS_INFO("Compute disparity");
    // bm->compute(img0.data,img1.data,disp); //compute disparity map
    // ROS_INFO("disparity computed");


    sensor_msgs::ChannelFloat32 depth_of_points;
    ROS_INFO("Copy features");

    //copy feature points
    sensor_msgs::PointCloud feature_points_depth;
    feature_points_depth.channels = feature_points->channels;
    feature_points_depth.header = feature_points->header;
    feature_points_depth.points = feature_points->points;

    //run BRIEF descriptor on Image 1
    cv::Mat desc_img1;
    Ptr<BriefDescriptorExtractor> brief_img1 =  cv::xfeatures2d::BriefDescriptorExtractor::create(64); //i don't know what 64 means - it was in the example
    brief_img1->compute(img1.data, im1_pts, desc_img1);

    //run BRIEF descriptor on Image 0
    cv::Mat desc_img0;
    Ptr<BriefDescriptorExtractor> brief_img0 =  cv::xfeatures2d::BriefDescriptorExtractor::create(64); //i don't know what 64 means - it was in the example
    brief_img0->compute(img0.data, feature_points_depth.points, desc_img0); //feature_points_depth.points might need to be of type vector<KeyPoint>

    // match the points
    vector<DMatch> matches;
    Ptr<BFMatcher> desc_matcher = cv::BFMatcher::create(cv::NORM_L2, true); //choose NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. 
    desc_matcher->match(desc_img0, desc_img1, matches, Mat());
}

int main(){
    cout << "hello world\n";
    return 0;
}