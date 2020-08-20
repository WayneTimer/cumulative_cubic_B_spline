#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
backward::SignalHandling sh;
} // namespace backward

#include <cstdio>
#include <string>
#include <queue>
#include <vector>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <ros/package.h>
#include <sophus/se3.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <interactive_markers/interactive_marker_server.h>
#include <interactive_markers/menu_handler.h>
#include <rviz_visual_tools/rviz_visual_tools.h>

#include "utils.h"

template <typename T>
class Cumu_b_spline
{
public:
    const double deltaT = 0.05;  // 50ms
    std::vector<Sophus::SE3Group<T>, Eigen::aligned_allocator<Sophus::SE3Group<T> > > SE3;

    Cumu_b_spline()
    {
        SE3.clear();
    }

    void add(const Eigen::Matrix<T, 3, 1> &translation,
             const Eigen::Quaternion<T> &quat)  // Eigen::Quaternion (w,x,y,z)
    {
        SE3.push_back(Sophus::SE3Group<T>(quat.toRotationMatrix(), translation));
        if (SE3.size() > 4)
        {
            SE3.erase(SE3.begin());
        }
    }

    void evaluate(const double &t,          // t \in [0, 1]
                  Eigen::Matrix<T, 3, 1> &translation, Eigen::Quaternion<T> &quat,
                  Eigen::Matrix<T, 3, 1> &linear_vel,
                  Eigen::Matrix<T, 3, 1> &acc, Eigen::Matrix<T, 3, 1> &omega)
    {
        if (SE3.size() < 4) return;

        std::vector<Eigen::Matrix<T, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<T, 4, 4> > > A,dA,ddA;
        A.resize(4);
        dA.resize(4);
        ddA.resize(4);

        Sophus::SE3Group<T> RTl0 = SE3[0];

        // ---- construct B ----
        Eigen::Matrix<T, 4, 4> B;
        B.setZero();
        B(0,0) = T(6.0);
        B(1,0) = T(5.0);
        B(1,1) = T(3.0);
        B(1,2) = T(-3.0);
        B(1,3) = T(1.0);
        B(2,0) = T(1.0);
        B(2,1) = T(3.0);
        B(2,2) = T(3.0);
        B(2,3) = T(-2.0);
        B(3,3) = T(1.0);

        Eigen::Matrix<T, 4, 4> tmp_B;
        tmp_B = T(1.0/6.0) * B;
        B = tmp_B;
        // --------------------

        Eigen::Matrix<T, 4, 1> T1( T(1.0), t, t*t, t*t*t);
        Eigen::Matrix<T, 4, 1> T2;
        T2 = B * T1;

        Eigen::Matrix<T, 4, 1> dT1( T(0.0), T(1.0), T(2.0)*t, T(3.0)*t*t);
        Eigen::Matrix<T, 4, 1> dT2;
        dT2 = T(1.0/deltaT) * B * dT1;

        Eigen::Matrix<T, 4, 1> ddT1( T(0.0), T(0.0), T(2.0), T(6.0)*t);
        Eigen::Matrix<T, 4, 1> ddT2;
        ddT2 = T(1.0/(deltaT*deltaT)) * B * ddT1;

        for (int j = 1; j <= 3; j++)  // 0 to 2 ? 1 to 3 ?  :  j= 1 to 3, diff with <Spline-fusion>
        {
            Eigen::Matrix<T, 6, 1> upsilon_omega = Sophus::SE3Group<T>::log(SE3[j-1].inverse() * SE3[j]);
            Eigen::Matrix<T, 4, 4> omega_mat;  // 4x4 se(3)
            omega_mat.setZero();
            Eigen::Matrix<T, 3, 1> for_skew;
            for_skew = upsilon_omega.template block<3,1>(3,0);
            omega_mat.template block<3,3>(0,0) = skew(for_skew);
            omega_mat.template block<3,1>(0,3) = upsilon_omega.template block<3,1>(0,0);

            // calc A
            T B_select = T2(j,0);
            // \omega 4x4 = /omega 6x1
            //  [ w^ v]        [ v ]
            //  [ 0  0]        [ w ]
            //
            // while multiply a scalar, the same. (Ignore the last .at(3,3) 1)
            Sophus::SE3Group<T> tmp = Sophus::SE3Group<T>::exp(B_select * upsilon_omega);
            A[j] = tmp.matrix();

            // calc dA
            T dB_select = dT2(j,0);
            dA[j] = A[j] * omega_mat * dB_select;

            // calc ddA
            T ddB_select = ddT2(j,0);
            ddA[j] = dA[j] * omega_mat * dB_select + A[j] * omega_mat * ddB_select;
        }

        Eigen::Matrix<T, 4, 4> all;

        // get B-spline's R,T
        all = A[1] * A[2] * A[3];
        Eigen::Matrix<T, 4, 4> ret = RTl0.matrix() * all;

        Eigen::Matrix<T, 3, 3> R;
        R = ret.template block<3, 3>(0, 0);
        translation = ret.template block<3, 1>(0, 3);
        quat = Eigen::Quaternion<T>(R);

        // get B-spline's omega
        Eigen::Matrix<T, 4, 4> dSE;
        all = dA[1]*A[2]*A[3] + A[1]*dA[2]*A[3] + A[1]*A[2]*dA[3];

        dSE = RTl0.matrix() * all;

        Eigen::Matrix<T, 3, 3> skew_R;
        skew_R = R.transpose() * dSE.template block<3,3>(0,0);
        T wx,wy,wz;  // ? simple mean
        wx = (-skew_R(1,2) + skew_R(2,1)) / T(2.0);
        wy = (-skew_R(2,0) + skew_R(0,2)) / T(2.0);
        wz = (-skew_R(0,1) + skew_R(1,0)) / T(2.0);
        omega(0) = wx;
        omega(1) = wy;
        omega(2) = wz;

        // get velocity
        linear_vel = dSE.template block<3,1>(0,3);  // world frame velocity
        Eigen::Matrix<T, 3, 1> body_linear_vel = R.transpose() * linear_vel;  // transform world frame vel to body frame

        // get B-spline's acc
        Eigen::Matrix<T, 4, 4> ddSE;
        all =   ddA[1]*A[2]*A[3] + A[1]*ddA[2]*A[3] + A[1]*A[2]*ddA[3]
              + T(2.0)*dA[1]*dA[2]*A[3] + T(2.0)*dA[1]*A[2]*dA[3] + T(2.0)*A[1]*dA[2]*dA[3];
        ddSE = RTl0.matrix() * all;

        Eigen::Matrix<T, 3, 1> spline_acc = R.transpose() * ddSE.template block<3,1>(0,3);  // ? g0 has been removed
        acc(0) = spline_acc(0, 0);
        acc(1) = spline_acc(1, 0);
        acc(2) = spline_acc(2, 0);
    }
};

Cumu_b_spline<double> cumu_bspline;
rviz_visual_tools::RvizVisualToolsPtr visual_tools_;

void goal_callback(const geometry_msgs::PoseStampedConstPtr &goal)
{
    Eigen::Vector3d translation(goal->pose.position.x,
                                goal->pose.position.y,
                                goal->pose.position.z);
    Eigen::Quaterniond quat(goal->pose.orientation.w,
                            goal->pose.orientation.x,
                            goal->pose.orientation.y,
                            goal->pose.orientation.z);
    ROS_INFO("Add pose: (x, y, z) = (%.2lf, %.2lf, %.2lf), (w, x, y, z) = (%.2lf, %.2lf, %.2lf, %.2lf)",
              translation(0), translation(1), translation(2),
              quat.w(), quat.x(), quat.y(), quat.z()
            );

    visual_tools_->publishAxis(goal->pose, 0.5, 0.1);

    cumu_bspline.add(translation, quat);

    // --- evaluate ---
    if (cumu_bspline.SE3.size() >= 4)
    {
        for (double ts = 0.0; ts < 1.0; ts+= 0.01)
        {
            Eigen::Vector3d translation, linear_vel, acc, omega;
            Eigen::Quaterniond quat;
            cumu_bspline.evaluate(ts, translation, quat, linear_vel, acc, omega);
            geometry_msgs::Pose pose;
            pose.position.x = translation(0);
            pose.position.y = translation(1);
            pose.position.z = translation(2);
            pose.orientation.w = quat.w();
            pose.orientation.x = quat.x();
            pose.orientation.y = quat.y();
            pose.orientation.z = quat.z();
            visual_tools_->publishAxis(pose);
        }
    }
    visual_tools_->trigger();
}

int main(int argc, char **argv)
{
    ros::init(argc,argv,"demo");
    ros::NodeHandle nh("~");

    ros::Publisher pub_path = nh.advertise<nav_msgs::Path>("path", 1000);

    // ---- subscribe ----
    ros::Subscriber sub_goal = nh.subscribe("/move_base_simple/goal", 10, goal_callback);
    // ------------------------

    visual_tools_.reset(new rviz_visual_tools::RvizVisualTools("map", "/pose_input"));

    while (ros::ok())
    {
        ros::spinOnce();
    }

    ros::shutdown();

    return 0;
}