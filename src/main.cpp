#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
backward::SignalHandling sh;
} // namespace backward

//#define M100

#include <cstdio>
#include <string>
#include <queue>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Imu.h>
#include <sophus/se3.hpp>
#include <boost/thread.hpp>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include "utils.h"

using namespace std;

queue<sensor_msgs::Imu> imu_buffer;
queue<nav_msgs::Odometry> odom_buffer;
boost::mutex mtx;
Eigen::Matrix4d B;
vector<Sophus::SE3d> SE3_vec;
vector<double> pose_ts_vec;
vector<double> imu_ts_vec;
FILE *file;  // B-spline: ts p \theta
FILE *omega_file;  // B-spline': ts \omega
FILE *vel_file;  // B-spline': ts vel
FILE *acc_file;  // B-spline'': ts acc
FILE *debug_file;  // VINS: ts p \theta vel
FILE *debug_imu_file;  // IMU: ts acc \omega
#ifdef M100
    double deltaT = 0.067;  // 15HZ pose (66.67ms)
#else
    double deltaT = 0.1;  // 10HZ img (100ms)
#endif


void imu_callback(const sensor_msgs::ImuConstPtr& msg)
{
    mtx.lock();
    imu_buffer.push(*msg);
    mtx.unlock();
}

void pose_callback(const geometry_msgs::PoseStampedConstPtr& msg)
{
    nav_msgs::Odometry odom_msg;
    odom_msg.header = msg->header;
    odom_msg.pose.pose = msg->pose;

    mtx.lock();
    odom_buffer.push(odom_msg);
    mtx.unlock();
}

void odom_callback(const nav_msgs::OdometryConstPtr& msg)
{
    mtx.lock();
    odom_buffer.push(*msg);
    mtx.unlock();
}

void spin_thread()
{
    while (ros::ok())
    {
        ros::spinOnce();
    }
}

void init()
{
    while (!imu_buffer.empty()) imu_buffer.pop();
    while (!odom_buffer.empty()) odom_buffer.pop();
    SE3_vec.clear();
    pose_ts_vec.clear();
    imu_ts_vec.clear();
    file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_pose.txt","w");
    omega_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_omega.txt","w");
    vel_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_vel.txt","w");
    acc_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_acc.txt","w");
    debug_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/debug_gt_pose.txt","w");
    debug_imu_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/debug_imu.txt","w");
    // --------
    B.setZero();

    B(0,0) = 6.0;

    B(1,0) = 5.0;
    B(1,1) = 3.0;
    B(1,2) = -3.0;
    B(1,3) = 1.0;

    B(2,0) = 1.0;
    B(2,1) = 3.0;
    B(2,2) = 3.0;
    B(2,3) = -2.0;

    B(3,3) = 1.0;

    Eigen::Matrix4d tmp_B;
    tmp_B = 1.0/6.0 * B;
    B = tmp_B;
    cout << B << endl;
}

// ensure there is enough msg
void process()
{
    ros::Time start_time_stamp = odom_buffer.front().header.stamp;
    while (!odom_buffer.empty())
    {
        ros::Time odom_stamp;
        geometry_msgs::Pose pose;
        Eigen::Vector3d linear_vel;
        odom_stamp = odom_buffer.front().header.stamp;
        pose = odom_buffer.front().pose.pose;
        linear_vel(0) = odom_buffer.front().twist.twist.linear.x;
        linear_vel(1) = odom_buffer.front().twist.twist.linear.y;
        linear_vel(2) = odom_buffer.front().twist.twist.linear.z;
        pose_ts_vec.push_back((odom_stamp - start_time_stamp).toSec());
        odom_buffer.pop();

        Eigen::Quaterniond q;
        q.w() = pose.orientation.w;
        q.x() = pose.orientation.x;
        q.y() = pose.orientation.y;
        q.z() = pose.orientation.z;
        Eigen::Vector3d p;
        p(0) = pose.position.x;
        p(1) = pose.position.y;
        p(2) = pose.position.z;
        Sophus::SE3d RT(q.toRotationMatrix(),p);
        SE3_vec.push_back(RT);

        Eigen::Matrix3d tmp_R = q.toRotationMatrix();
        Eigen::Vector3d theta = R_to_ypr(tmp_R);
        fprintf(debug_file,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                            (odom_stamp - start_time_stamp).toSec(),
                            p(0),p(1),p(2),
                            theta(0),theta(1),theta(2),
                            linear_vel(0),linear_vel(1),linear_vel(2)
               );
    }
    while (!imu_buffer.empty() && imu_buffer.front().header.stamp < start_time_stamp)
    {
        imu_buffer.pop();
    }
    while (!imu_buffer.empty())
    {
        sensor_msgs::Imu imu;
        imu = imu_buffer.front();
        imu_buffer.pop();
        imu_ts_vec.push_back((imu.header.stamp - start_time_stamp).toSec());

        fprintf(debug_imu_file,"%lf %lf %lf %lf %lf %lf %lf\n",
                                (imu.header.stamp - start_time_stamp).toSec(),
                                imu.linear_acceleration.x,imu.linear_acceleration.y,imu.linear_acceleration.z,
                                imu.angular_velocity.x,imu.angular_velocity.y,imu.angular_velocity.z
               );
    }
    //---------------------
    double diff = 0.001;  // 1ms intepolate
    int l;
    l = SE3_vec.size();
    for (int i=1;i<l-2;i++)
    {
        Sophus::SE3d RTl0 = SE3_vec[i-1];
        for (double ts=pose_ts_vec[i];ts<pose_ts_vec[i+1];ts+=diff)
        {
            double u = (ts-pose_ts_vec[i])/deltaT;
            Eigen::Vector4d T1(1.0,u,u*u,u*u*u);
            Eigen::Vector4d T2;
            T2 = B*T1;

            vector<Sophus::SE3d> tmp_SE;
            tmp_SE.resize(4);
            for (int k=1;k<=3;k++)
            {
                int j = i-1+k;
                Eigen::VectorXd upsilon_omega = Sophus::SE3d::log(SE3_vec[j-1].inverse() * SE3_vec[j]);
                double B_select = T2(k); // B_j(u) -> B_k(u) ?  TUM's paper   k?   k-1?  :   k
                tmp_SE[k] = Sophus::SE3d::exp(B_select * upsilon_omega);
            }
            Sophus::SE3d res;
            res = tmp_SE[1] * tmp_SE[2] * tmp_SE[3];
            Sophus::SE3d ret = RTl0 * res;

            Eigen::Vector3d T,theta;
            Eigen::Matrix3d R;
            T = ret.translation();
            R = ret.rotationMatrix();
            theta = R_to_ypr(R);
            fprintf(file,"%lf %lf %lf %lf %lf %lf %lf\n",
                          ts,
                          T(0),T(1),T(2),                                           
                          theta(0),theta(1),theta(2)
                   );

            // get omega,acc
            vector<Eigen::Matrix4d> A,dA,ddA;
            A.resize(4);
            dA.resize(4);
            ddA.resize(4);
            for (int j=1;j<=3;j++)  // 0 to 2 ? 1 to 3 ?  :  j= 1 to 3, diff with <Spline-fusion>
            {
                // calc A
                Eigen::VectorXd upsilon_omega = Sophus::SE3d::log(SE3_vec[i+j-2].inverse() * SE3_vec[i+j-1]);
                double B_select = T2(j);
                // \omega 4x4 = /omega 6x1
                //  [ w^ v]        [ v ]
                //  [ 0  0]        [ w ]
                //
                // while multiply a scalar, the same. (Ignore the last .at(3,3) 1)
                A[j] = (Sophus::SE3d::exp(B_select * upsilon_omega)).matrix();

                Eigen::Matrix4d omega_mat;  // 4x4 se(3)
                omega_mat.setZero();
                omega_mat.block<3,3>(0,0) = skew(upsilon_omega.tail<3>());
                omega_mat.block<3,1>(0,3) = upsilon_omega.head<3>();

                // calc dA
                Eigen::Vector4d dT1(0.0,1.0,2.0*u,3.0*u*u);
                Eigen::Vector4d dT2;
                dT2 = 1.0/deltaT * B * dT1;
                double dB_select = dT2(j);
                dA[j] = A[j] * omega_mat * dB_select;

                // calc ddA
                Eigen::Vector4d ddT1(0.0,0.0,2.0,6.0*u);
                Eigen::Vector4d ddT2;
                ddT2 = 1.0/(deltaT*deltaT) * B * ddT1;
                double ddB_select = ddT2(j);
                ddA[j] = dA[j] * omega_mat * dB_select + A[j] * omega_mat * ddB_select;
            }

            Eigen::Matrix4d all;
            // get B-spline's omega
            Eigen::Matrix4d dSE;
            all = dA[1]*A[2]*A[3] + A[1]*dA[2]*A[3] + A[1]*A[2]*dA[3];

            dSE = RTl0.matrix() * all;

            Eigen::Matrix3d skew_R = ret.rotationMatrix().transpose() * dSE.block<3,3>(0,0);
            double wx,wy,wz;  // ? simple mean
            wx = (-skew_R(1,2) + skew_R(2,1)) / 2.0;
            wy = (-skew_R(2,0) + skew_R(0,2)) / 2.0;
            wz = (-skew_R(0,1) + skew_R(1,0)) / 2.0;
            Eigen::Vector3d linear_vel = dSE.block<3,1>(0,3);  // world frame velocity

            fprintf(omega_file,"%lf %lf %lf %lf\n",
                              ts,
                              wx,wy,wz
                   );
            fprintf(vel_file,"%lf %lf %lf %lf\n",
                              ts,
                              linear_vel(0),linear_vel(1),linear_vel(2)
                   );

            // get B-spline's acc
            Eigen::Matrix4d ddSE;
            all =   ddA[1]*A[2]*A[3] + A[1]*ddA[2]*A[3] + A[1]*A[2]*ddA[3]
                  + 2.0*dA[1]*dA[2]*A[3] + 2.0*dA[1]*A[2]*dA[3] + 2.0*A[1]*dA[2]*dA[3];
            ddSE = RTl0.matrix() * all;

            Eigen::Vector3d spline_acc = ret.rotationMatrix().transpose() * (ddSE.block<3,1>(0,3) + Eigen::Vector3d(0,0,9.805));  // ? gravity not accurate

            fprintf(acc_file,"%lf %lf %lf %lf\n",
                              ts,spline_acc(0),spline_acc(1),spline_acc(2)
                   );
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc,argv,"Dense_Tracking");
    ros::NodeHandle nh("~");

    ros::Subscriber sub_pose = nh.subscribe("/self_calibration_estimator/pose",1000,pose_callback);  // 10HZ (100ms)
#ifdef M100
    ros::Subscriber sub_imu = nh.subscribe("/djiros/imu",1000,imu_callback);  // 100HZ (10ms)
    ros::Subscriber sub_odom = nh.subscribe("/self_calibration_estimator/odometry",1000,odom_callback);
#else
    ros::Subscriber sub_imu = nh.subscribe("/imu0",1000,imu_callback);  // 200HZ (5ms)
#endif

    init();

    boost::thread th1(spin_thread);

    getchar();
    mtx.lock();
    process();
    fclose(file);
    fclose(omega_file);
    fclose(vel_file);
    fclose(acc_file);
    fclose(debug_file);
    fclose(debug_imu_file);
    mtx.unlock();
    ros::shutdown();

    return 0;
}
