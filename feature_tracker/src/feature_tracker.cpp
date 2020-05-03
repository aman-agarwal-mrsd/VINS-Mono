#include "feature_tracker.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/mat.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// Reduces the current vector according to the status array, we only keep elements in the vector v
// If status is true else we remove the points and reduce the size of the vector
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if(FISHEYE) // Euroc config not using fisheye
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    // if there is no forw_img (i.e. first image)
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        //computes the optical flow of the features from cur_img to forw_img
        //forw_pts hold the output 2D point coordinates of the calculated new positions of input features in the second image
        //status vector of length of features that holds 1 if corresponding feature found and 0 if not for every feature
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
        
        // Removes points for which optical flow was not there or they were outiside boundary of image
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        // Reduces the vectors according to the status count 
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            //detect features to be tracked in new image aka forw_img
            //function is used to init a point-based tracker of an object
            //n_pts holds vector of detected corner points
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        // Adds the n_pts to the tracker object
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    //update variables from new image
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        //determine fundamental matrix between features in first image and features in second image
        //using RANSAC method
        //
        //status is a mask that holds 0 for outliers and 1 for other points
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

// Reads the camera calibration matrix for the respective camera from the YAML file
void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}


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
    // vector<cv::Point2f> im1_pts;// I think this needs to be of type vector<KeyPoint>
    // setMask();
    // cv::goodFeaturesToTrack(forw_img, im1_pts, feature_points->points.size(), 0.01, MIN_DIST, mask);


    //bm->setSpeckleRange(32);
    ROS_INFO("Compute disparity");
    bm->compute(img0.data,img1.data,disp); //compute disparity map
    ROS_INFO("disparity computed");


    sensor_msgs::ChannelFloat32 depth_of_points;
    ROS_INFO("Copy features");

    //copy feature points
    sensor_msgs::PointCloud feature_points_depth;
    feature_points_depth.channels = feature_points->channels;
    feature_points_depth.header = feature_points->header;
    feature_points_depth.points = feature_points->points;

    //run BRIEF descriptor on Image 1
    // cv::Mat desc_img1;
    // Ptr<BriefDescriptorExtractor> brief_img1 =  cv::xfeatures2d::BriefDescriptorExtractor::create(64); //i don't know what 64 means - it was in the example
    // brief_img1->compute(img1.data, im1_pts, desc_img1);

    //run BRIEF descriptor on Image 0
    // cv::Mat desc_img0;
    // Ptr<BriefDescriptorExtractor> brief_img0 =  cv::xfeatures2d::BriefDescriptorExtractor::create(64); //i don't know what 64 means - it was in the example
    // brief_img0->compute(img0.data, feature_points_depth.points, desc_img0); //feature_points_depth.points might need to be of type vector<KeyPoint>

    // match the points
    // vector<DMatch> matches;
    // Ptr<BFMatcher> desc_matcher = cv::BFMatcher::create(cv::NORM_L2, true); //choose NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. 
    // desc_matcher->match(desc_img0, desc_img1, matches, Mat());



    ROS_INFO("About to get depth values");
    for (unsigned int j = 0; j<feature_points->points.size(); j++)
    {
        int x = feature_points->points[j].x;
        int y = feature_points->points[j].y;
        
        float depth = disp.at<double>(x,y);

        depth_of_points.values.push_back(depth);
    }

    feature_points_depth.channels.push_back(depth_of_points); //this is index 5 i think

    return feature_points_depth;

}

sensor_msgs::ChannelFloat32 FeatureTracker::computeDepthMap2(const cv::Mat &_img0, const cv::Mat &_img1, const vector<cv::Point2f> &feature_points)
{
    ROS_INFO("Computing Depth Map");

    cv::Mat img0, img1;
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img0, img0);
        clahe->apply(_img1, img1);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img0 = _img0;
        img1 = _img1;
    
    sensor_msgs::ChannelFloat32 depth_of_point;
    depth_of_point.name = "Depth";

    //Account for distortion in intrinsics
    float fx = 4.6115862106007575e+02, fy = 4.5975286598073296e+02, cx = 3.6265929181685937e+02, cy = 2.4852105668448124e+02;
    Mat intrinsic_cam0 = (Mat1d(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    float k1 = -2.9545645106987750e-01, k2 = 8.6623215640186171e-02, p1 = 2.0132892276082517e-06, p2 = 1.3924531371276508e-05;
    Mat distortion_coefficients_cam0 = (Mat1d(1, 4) << k1, k2, p1, p2); 

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

    // Creating final projection matrixes for both cameras  
    cv::Mat cam0_proj = new_intrinsic_cam0 * cam0_to_cam0;
    cv::Mat cam1_proj = new_intrinsic_cam1 * cam1_to_cam0;

    // Find features for img1
    vector<Point2f> img0_features;
    vector<KeyPoint> img0_kps;
    int max_corners = 150;
    double quality = 0.01, min_distance = 30;   
    vector<Point2f> img1_features;
    vector<KeyPoint> img1_kps;
    goodFeaturesToTrack(img1,img1_features,max_corners,quality,min_distance);
    for (int i=0; i<img1_features.size();i++) {
        Point2f pt_to_push1 = img1_features[i];
        KeyPoint img1_kp;
        img1_kp.pt = pt_to_push1;
        img1_kps.push_back(img1_kp);
    }

    // Copy img0 features to different container
    vector<KeyPoint> img0_kps;
    for (int i=0; i<feature_points.size();i++) {
        Point2f pt_to_push0 = feature_points[i];
        KeyPoint img0_kp;
        img0_kp.pt = pt_to_push0;
        img0_kps.push_back(img0_kp);
    }

    //run BRIEF descriptor on Image 0
    cv::Mat img0_desc;
    Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief_img0 =  cv::xfeatures2d::BriefDescriptorExtractor::create(64); //i don't know what 64 means - it was in the example
    brief_img0->compute(img0, img0_kps, img0_desc); //feature_points_depth.points might need to be of type vector<KeyPoint>

    //run BRIEF descriptor on Image 1
    cv::Mat img1_desc;
    Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief_img1 =  cv::xfeatures2d::BriefDescriptorExtractor::create(64); //i don't know what 64 means - it was in the example
    brief_img1->compute(img1, img1_kps, img1_desc); //feature_points_depth.points might need to be of type vector<KeyPoint>

    // Find correspondances
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
    }

    cv::Mat pnts3D;// Output Matrix

    cv::triangulatePoints(cam0_proj,cam1_proj,triangulation_points0,triangulation_points1,pnts3D);
    
    return depth_of_point;
}