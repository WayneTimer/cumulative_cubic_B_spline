#ifndef data_structure_h
#define data_structure_h

#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>

#include "parameters.h"

using namespace std;

class IMU_DATA
{
public:
    double ts;
    Eigen::Vector3d a,w;

    IMU_DATA(double _ts, Eigen::Vector3d _a, Eigen::Vector3d _w)
            :ts(_ts),a(_a),w(_w)
    {}
};

class State // every frame is a state
{
public:
    Eigen::Vector3d p,gt_p; // p_k^0
    Eigen::Quaterniond q,gt_q; // R_k^0
    Eigen::Vector3d v; // v_k^k (only use in B-spline.cpp yet)
    Eigen::Matrix3d R_k1_k; // integrate imu_omega between (k,k+1) -> R_k1^k, store in k+1 frame
    Eigen::Vector3d alpha; // alpha_k1^k, store in k+1 frame
    Eigen::Vector3d beta; // beta_k1^k, store in k+1 frame
    cv::Mat img; // key frame? img from camera : estimated img
    Eigen::MatrixXd img_data[PYRDOWN_LEVEL]; // img from camera
    Eigen::MatrixXd depth[PYRDOWN_LEVEL]; // SGBM from stereo camera
    int depth_cnt; // cnt of (img depth > threshold)
    ros::Time ros_stamp;    // raw ros_stamp
    double stamp;  // (ros_stamp-start_time_stamp).toSec()
    double DeltaT;
    vector<IMU_DATA> imu_data;  // camera body frame, without g0 & bias: ros_stamp-start_time_stamp, a, w

    State();
};

class Graph
{
public:
    vector<State> state;

    Graph();
};

class CALI_PARA
{
public:
    double fx[PYRDOWN_LEVEL],fy[PYRDOWN_LEVEL],cx[PYRDOWN_LEVEL],cy[PYRDOWN_LEVEL];
    double baseline;
    int width[PYRDOWN_LEVEL],height[PYRDOWN_LEVEL];
    Eigen::Matrix3d R_I_2_C; // R_I^C
    Eigen::Vector3d T_I_2_C; // T_I^C

    double exposure_time; // exposure_time in second
    int img_hz;

    bool init(ros::NodeHandle& nh);
    void view();
};
#endif
