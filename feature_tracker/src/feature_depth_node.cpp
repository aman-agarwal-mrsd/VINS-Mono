#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
// #include <cv_bridge/cv_bridge.h>
// #include <opencv2/opencv.hpp>
// #include <ros/ros.h>
#include <sensor_msgs/Image.h>
// #include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
// #include <sensor_msgs/Imu.h>
// #include "feature_tracker.h"
// #include "estimator.h"
// #include "parameters.h"
// #include "utility/visualization.h"

//Initialize main estimator object
// Estimator estimator;

// std::condition_variable con;
// double current_time = -1;
// queue<sensor_msgs::ImuConstPtr> imu_buf;
using namespace std;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex f_buf;
std::mutex i0_buf;
std::mutex i1_buf;
double latest_time = 0;
int max_queue_size = 100;
int sum_of_wait = 0;
std::mutex i_buf;

std::mutex m_state;

std::mutex m_estimator;


// Eigen::Vector3d tmp_P;
// Eigen::Quaterniond tmp_Q;
// Eigen::Vector3d tmp_V;
// Eigen::Vector3d tmp_Ba;
// Eigen::Vector3d tmp_Bg;
// Eigen::Vector3d acc_0;
// Eigen::Vector3d gyr_0;
// bool init_feature = 0;
// bool init_imu = 1;
// double last_imu_t = 0;


struct pcl_images_map
{
    sensor_msgs::PointCloudConstPtr feature_msg;
    sensor_msgs::ImageConstPtr img0_msg;
    sensor_msgs::ImageConstPtr img1_msg;

    pcl_images_map(const sensor_msgs::PointCloudConstPtr &_feature_msg,
    const sensor_msgs::ImageConstPtr &_img0_msg,
    const sensor_msgs::ImageConstPtr &_img1_msg
    ): feature_msg(_feature_msg), img0_msg(_img0_msg), img1_msg(img1_msg){}
};
typedef struct pcl_images_map pcl_images_map;


//getMeasurements function called in process function
std::vector<pcl_images_map> getMeasurements()
{
    std::vector<pcl_images_map> measurements;

    while (true)
    {

        // ROS_INFO("IMG0 Buffer Size %d", img0_buf.size());
        // ROS_INFO("IMG1 Buffer Size %d", img1_buf.size());
        // ROS_INFO("Feature Buffer Size %d", feature_buf.size());

        // If no PCL or Left Image then we return
        if (img0_buf.empty() || feature_buf.empty()||img1_buf.empty())
        {   
            // ROS_INFO("One of the buffer is empty");
            return measurements;
        }
        // Left Camera Images are older than current PCL
        if (img0_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec())
        {
            // ROS_INFO("Left camera has extra images");
            img0_buf.pop();
        }
        // The left camera images are newer or equal to PCL
        else
        {
            // Right Camera image is newer than or equal to PCL 
            if (feature_buf.front()->header.stamp.toSec() <= img1_buf.front()->header.stamp.toSec())
            {
                ROS_INFO("Pushing to measurement");
                pcl_images_map m(feature_buf.front(),img0_buf.front(), img1_buf.front());
                measurements.emplace_back(m);
                ROS_INFO("Pushed to measurement");
                feature_buf.pop();
                img0_buf.pop();
                img1_buf.pop();
            }
            // The left camera images are newer or equal to PCL
            else
            {
                img1_buf.pop();
            }
            
        }
    }
    return measurements;
}








// thread: Depth Estimation
void depth_estimator()
{
    // ROS_INFO("INSIDE DEPTH ESTIMATOR");
    // return;
    while (true)
    {
        // std::vector<pcl_images_map> measurements;
        // std::unique_lock<std::mutex> lk(m_buf);
        // con.wait(lk, [&]
        //          {
        //              // Populate measurements with vector of imu readings and vector of point clouds
        //     return (measurements = getMeasurements()).size() != 0;
        //          });
        // lk.unlock();
        // m_estimator.lock();
        // ROS_INFO("Depth Estimator");
        std::vector<pcl_images_map> measurements = getMeasurements();
        // Returns if measurements was empty
        if (measurements.empty())
        {
            // ROS_INFO("measurement empty");
            return;
        }
        int m_size = measurements.size();
        ROS_INFO("Measurement size: %d", m_size);
        // Measurements was not emptyNFO
        for (auto &measurement : measurements)
        {
            ROS_INFO(" Data type: %s", typeid(measurement).name());
            ROS_INFO("INSIDE FOR LOOP");
            // auto img0_msg = measurement.img0_msg; // point cloud measurements
            // auto img1_msg = measurement.img1_msg; // point cloud measurements
            // auto feature_msg = measurement.feature_msg; // point cloud measurements
            ROS_INFO("Left Image Time: %f", measurement.img0_msg->header.stamp.toSec());
            ROS_INFO("Right Image Time: %f", measurement.img1_msg->header.stamp.toSec());
            ROS_INFO("Feature Time: %f", measurement.feature_msg->header.stamp.toSec());
            // cout<< "Left Image Time: "<<img0_msg->header.stamp.toSec()<< " Right Image Time: "<<img1_msg->header.stamp.toSec()<<
            // "Feature Time: "<<feature_msg->header.stamp.toSec()<<endl;
        }
    }
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    f_buf.lock();
    if (feature_buf.size() > max_queue_size)
    {
        feature_buf.pop();
    }   

    feature_buf.push(feature_msg); // puts new data into feature_buf which is of type queue<sensor_msgs::PointCloudConstPtr> 
    f_buf.unlock();
    // con.notify_one();
    return;
}

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    //printf("relocalization callback! \n");
    //stores point cloud and handeled in process function below?
    i0_buf.lock();
    if (img0_buf.size() > max_queue_size)
    {
        img0_buf.pop();
    }

    img0_buf.push(img_msg);
    i0_buf.unlock();
    return;
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    //printf("relocalization callback! \n");
    //stores point cloud and handeled in process function below?
            i1_buf.lock();
    if (img1_buf.size() > max_queue_size)
    {
        img1_buf.pop();
    }
        img1_buf.push(img_msg);
        i1_buf.unlock();
    
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_depth");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    // readParameters(n);
    // estimator.setParameter();

    // registerPub(n);


    ros::Subscriber sub_img0 = nh.subscribe("/cam0/image_raw", 100, img0_callback);
    ros::Subscriber sub_img1 = nh.subscribe("/cam1/image_raw", 100, img1_callback);
    ros::Subscriber sub_feature = nh.subscribe("/feature_tracker/feature", 2000, feature_callback);
    // ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Rate loop_rate(10);
    while(ros::ok())
    {
        // ROS_INFO("INSIDE WHILE");
        depth_estimator();
        ros::spinOnce();
        // std::thread measurement_process{process};
        // ros::spin();
    }
    // ros::spin();

    return 0;
}
