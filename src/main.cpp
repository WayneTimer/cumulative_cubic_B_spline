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
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Imu.h>
#include <sophus/se3.hpp>
#include <boost/thread.hpp>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include "utils.h"
#include "ceres_solve.h"

using namespace std;

Graph graph;
CALI_PARA cali;

queue<sensor_msgs::Image> img1_buffer;
queue<stereo_msgs::DisparityImage> disp_buffer;
queue<sensor_msgs::Imu> imu_buffer;
boost::mutex mtx;
Eigen::Matrix4d B;
FILE *file;  // B-spline: ts p \theta
FILE *omega_file;  // B-spline': ts \omega
FILE *vel_file;  // B-spline': ts vel
FILE *acc_file;  // B-spline'': ts acc
FILE *debug_file;  // VINS: ts p \theta vel
FILE *debug_imu_file;  // IMU: ts acc \omega
double deltaT = 0.05;  // 20HZ img (50ms)
nav_msgs::Path path;
Eigen::Vector3d g0,initial_omega_bias;
double start_time_stamp,last_imu_stamp;
int key_frame_no;

void ros_pub_points(State& state, ros::Publisher& pub_pc2, ros::Time& ros_stamp)
{
    int w,h,level;
    level = 0;
    w = state.depth[level].cols();
    h = state.depth[level].rows();
    

    sensor_msgs::PointCloud2 pc2_msg;
    pc2_msg.header.stamp = ros_stamp;
    pc2_msg.header.frame_id = "ref_frame";
    pc2_msg.height = h;
    pc2_msg.width = w;
    pc2_msg.fields.resize(4);
    pc2_msg.fields[0].name = "x";
    pc2_msg.fields[0].offset = 0;
    pc2_msg.fields[0].count = 1;
    pc2_msg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    pc2_msg.fields[1].name = "y";
    pc2_msg.fields[1].offset = 4;
    pc2_msg.fields[1].count = 1;
    pc2_msg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    pc2_msg.fields[2].name = "z";
    pc2_msg.fields[2].offset = 8;
    pc2_msg.fields[2].count = 1;
    pc2_msg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    pc2_msg.fields[3].name = "rgb";
    pc2_msg.fields[3].offset = 12;
    pc2_msg.fields[3].count = 1;
    pc2_msg.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
    pc2_msg.is_bigendian = false;
    pc2_msg.point_step = sizeof(float) * 4;
    pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width;
    pc2_msg.data.resize(pc2_msg.row_step * pc2_msg.height);
    pc2_msg.is_dense = true;

    int i = 0;
    for (int u=0;u<h;u++)
        for (int v=0;v<w;v++,i++)
        {
            Eigen::Vector3d p_ref;
            double lambda;
            lambda = state.depth[level](u,v);

            if (double_equ_check(lambda,0.0,DOUBLE_EPS)<=0) // no depth
                lambda = 0.0;

            p_ref = Eigen::Vector3d( (v-cali.cx[level])/cali.fx[level],
                                     (u-cali.cy[level])/cali.fy[level],
                                      1.0) * lambda; // (x,y,z)^{ref}

            float x = p_ref(0);
            float y = p_ref(1);
            float z = p_ref(2);
            uchar g = (uchar)state.img_data[level](u,v);
            int32_t rgb = (g << 16) | (g << 8) | g;

            memcpy(&pc2_msg.data[i * pc2_msg.point_step + 0], &x, sizeof(float));
            memcpy(&pc2_msg.data[i * pc2_msg.point_step + 4], &y, sizeof(float));
            memcpy(&pc2_msg.data[i * pc2_msg.point_step + 8], &z, sizeof(float));
            memcpy(&pc2_msg.data[i * pc2_msg.point_step + 12], &rgb, sizeof(int32_t));
        }
    pub_pc2.publish(pc2_msg);
}

// pyr_down: (img, depth)
void pyr_down(State& state,int level)
{
    if (level==0) return;

    int w,h;
    w = cali.width[level];
    h = cali.height[level];
    state.img_data[level] = Eigen::MatrixXd::Zero(h,w);
    state.depth[level] = Eigen::MatrixXd::Zero(h,w);
    printf("pyrdown: %d x %d\n",w,h);

    for (int u=0;u<w;u++)
        for (int v=0;v<h;v++)
        {
            int u1,v1;
            u1 = u<<1;
            v1 = v<<1;

            double t;
            t = ( state.img_data[level-1](v1,u1)
                + state.img_data[level-1](v1,u1+1)
                + state.img_data[level-1](v1+1,u1)
                + state.img_data[level-1](v1+1,u1+1))
                / 4.0;
            state.img_data[level](v,u) = t;

            t = ( state.depth[level-1](v1,u1)
                + state.depth[level-1](v1,u1+1)
                + state.depth[level-1](v1+1,u1)
                + state.depth[level-1](v1+1,u1+1))
                    / 4.0;
            state.depth[level](v,u) = t;
        }
}

void vision_callback(
    const sensor_msgs::ImageConstPtr& msg_img_l,
    const stereo_msgs::DisparityImageConstPtr& msg_disp)
{
    mtx.lock();
    img1_buffer.push(*msg_img_l);
    disp_buffer.push(*msg_disp);
    mtx.unlock();
}

void imu_callback(const sensor_msgs::ImuConstPtr& msg)
{
    mtx.lock();
    imu_buffer.push(*msg);
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
    file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_pose.txt","w");
    omega_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_omega.txt","w");
    vel_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_vel.txt","w");
    acc_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_acc.txt","w");
    debug_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/debug_gt_pose.txt","w");
    debug_imu_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/debug_imu.txt","w");
    // ---- construct B ----
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
    // ----------

    // === read config ===
    ros::NodeHandle nh("~");

    if (!cali.init(nh))
    {
        ROS_ERROR("cali init error!");
        exit(1);
    }
    cali.view();

    while (!imu_buffer.empty()) imu_buffer.pop();
    while (!img1_buffer.empty()) img1_buffer.pop();
    while (!disp_buffer.empty()) disp_buffer.pop();

    path.header.frame_id = "world";
    path.poses.clear();
}

void get_stationary_imu()
{
    ros::Time imu_stamp,img_stamp;

    // ensure there is IMG (& disp) in buffer
    mtx.lock();
    while (img1_buffer.empty())
    {
        mtx.unlock();
        usleep(1000);
        mtx.lock();
    }
    img_stamp = img1_buffer.front().header.stamp;
    mtx.unlock();

    // ensure there is IMU in the buffer, and this IMU stamp < IMG stamp
    mtx.lock();
    while (imu_buffer.empty() || imu_buffer.front().header.stamp < img_stamp)
    {
        if (!imu_buffer.empty()) imu_buffer.pop();
        mtx.unlock();
        usleep(1000);
        mtx.lock();
    }
    imu_stamp = imu_buffer.front().header.stamp;
    mtx.unlock();

    // ========================================
    //  drop first img & disp for computing g0
    // ========================================
    mtx.lock();
    img1_buffer.pop();
    disp_buffer.pop();
    while (img1_buffer.empty())
    {
        mtx.unlock();
        usleep(1000);
        mtx.lock();
    }
    img_stamp = img1_buffer.front().header.stamp;
    mtx.unlock();
    // -----
    g0.setZero();
    omega_bias.setZero();
    int cnt = 0;
    mtx.lock();
    while (imu_buffer.empty() || imu_buffer.front().header.stamp < img_stamp)
    {
        while (!imu_buffer.empty() && imu_buffer.front().header.stamp < img_stamp)
        {
            cnt++;
            g0(0) = g0(0) + imu_buffer.front().linear_acceleration.x;
            g0(1) = g0(1) + imu_buffer.front().linear_acceleration.y;
            g0(2) = g0(2) + imu_buffer.front().linear_acceleration.z;
            omega_bias(0) = omega_bias(0) + imu_buffer.front().angular_velocity.x;
            omega_bias(1) = omega_bias(1) + imu_buffer.front().angular_velocity.y;
            omega_bias(2) = omega_bias(2) + imu_buffer.front().angular_velocity.z;
            imu_buffer.pop();
        }
        mtx.unlock();
        usleep(1000);
        mtx.lock();
    }
    mtx.unlock();
    g0 = g0 / cnt;
    omega_bias = omega_bias / cnt;

    // ===== output g0, omega_bias, cnt =====
    cout << "g0: " << g0.transpose() << endl;
    cout << "omega_bias: " << omega_bias.transpose() << endl;
    cout << "cnt: " << cnt << endl;
}

void load_img_disp(State& state, cv::Mat &img)
{
    // load cv_img into Eigen matrix
    state.img_data[0] = Eigen::MatrixXd::Zero(cali.height[0],cali.width[0]);
    for (int u=0;u<img.rows;u++)
        for (int v=0;v<img.cols;v++)
            state.img_data[0](u,v) = img.at<uchar>(u,v);

    for (int level=0;level<PYRDOWN_LEVEL;level++)
    {
        pyr_down(state,level); // also pyr_down depth
        calc_gradient(state.img_data[level],
                      state.gx[level],
                      state.gy[level]);
    }
}

void real_data(State& state)
{
    cv::Mat img,disp;
    sensor_msgs::Image img_msg;
    stereo_msgs::DisparityImage disp_msg;

    // get IMG & disp
    mtx.lock();
    while (img1_buffer.empty() || disp_buffer.empty())
    {
        mtx.unlock();
        usleep(1000);
        mtx.lock();
    }
    img_msg = img1_buffer.front();
    disp_msg = disp_buffer.front();
    img1_buffer.pop();
    disp_buffer.pop();
    mtx.unlock();

    img = cv_bridge::toCvCopy(img_msg)->image;
    disp = cv_bridge::toCvCopy(disp_msg.image)->image;
    state.ros_stamp = img_msg.header.stamp;

    int depth_cnt;
    depth_cnt = cal_depth_img(disp,state.depth[0],cali.baseline,cali.fx[0]);  //float
    load_img_disp(state,img);

    if (graph.state.size()==0) // first state, only get IMG & disp & set p=(0,0,0)
    {
        start_time_stamp = img_msg.header.stamp.toSec();
        state.p.setZero();
        state.q.setIdentity();
        state.v.setZero();
        last_imu_stamp = 0.0;
        key_frame_no = 0;
    }
    else
    {
        // ==== IMU integrate ====
        int last_no = graph.state.size()-1;  // cur state not yet pushed

        double &DeltaT = state.DeltaT;
        Eigen::Matrix3d &R_k1_k = state.R_k1_k;
        Eigen::Vector3d &alpha = state.alpha;
        Eigen::Vector3d &beta = state.beta;
        Eigen::Vector3d &pk1 = state.p;
        Eigen::Vector3d &vk1 = state.v;
        Eigen::Quaterniond &qk1 = state.q;

        R_k1_k.setIdentity();
        alpha.setZero();
        beta.setZero();
        DeltaT = 0.0;

        mtx.lock();
        while (imu_buffer.empty() || imu_buffer.front().header.stamp < state.ros_stamp)
        {
            if (imu_buffer.empty())
            {
                mtx.unlock();
                usleep(1000);
                mtx.lock();
                continue;
            }
            sensor_msgs::Imu imu_msg;
            imu_msg = imu_buffer.front();
            imu_buffer.pop();
            mtx.unlock();

            double imu_t = imu_msg.header.stamp.toSec() - start_time_stamp;
            double dt = imu_t - last_imu_stamp;
            DeltaT += dt;
            Eigen::Vector3d a,w;

            a(0) = imu_msg.linear_acceleration.x;
            a(1) = imu_msg.linear_acceleration.y;
            a(2) = imu_msg.linear_acceleration.z;
            w(0) = imu_msg.angular_velocity.x - omega_bias(0);
            w(1) = imu_msg.angular_velocity.y - omega_bias(1);
            w(2) = imu_msg.angular_velocity.z - omega_bias(2);

            Eigen::Quaterniond dq;
            dq.x() = w(0)*dt*0.5;
            dq.y() = w(1)*dt*0.5;
            dq.z() = w(2)*dt*0.5;
            dq.w() = sqrt(1.0 - (dq.x()*dq.x()) - (dq.y()*dq.y()) - (dq.z()*dq.z()) );
            dq.normalize();

            Eigen::Matrix3d dR(dq);
            R_k1_k = R_k1_k * dR;

            // --- remove g0:   R_t^0 = R_k^0 * R_t^k ---
            Eigen::Matrix3d R_t_0;
            R_t_0 = graph.state[last_no].q.toRotationMatrix() * R_k1_k;
            a = a - R_t_0.transpose() * g0;
            // -------

            // ---- transform a,w to camera frame ----
            a = cali.R_I_2_C * a + cali.T_I_2_C;
            w = cali.R_I_2_C * w + cali.T_I_2_C;
            // -----

            alpha += beta*dt + 0.5*R_k1_k*a*dt*dt;
            beta += R_k1_k*a*dt;

            last_imu_stamp = imu_t;
        }
        printf("DeltaT = %lf\n",DeltaT);
        pk1 = graph.state[last_no].p 
            + graph.state[last_no].q.toRotationMatrix()*graph.state[last_no].v*DeltaT
            + graph.state[last_no].q.toRotationMatrix()*alpha; // g0 has already removed
        vk1 = R_k1_k.transpose()
            * (graph.state[last_no].v + beta);  // g0 has already removed
        qk1 = Eigen::Quaterniond(graph.state[last_no].q.toRotationMatrix() * R_k1_k);
        qk1.normalize();
    }
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

        Eigen::Vector3d tmp_acc(imu.linear_acceleration.x,imu.linear_acceleration.y,imu.linear_acceleration.z);
        imu_acc_vec.push_back(tmp_acc);
        Eigen::Vector3d tmp_omega(imu.angular_velocity.x,imu.angular_velocity.y,imu.angular_velocity.z);
        imu_omega_vec.push_back(tmp_omega);

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

            Eigen::Vector4d dT1(0.0,1.0,2.0*u,3.0*u*u);
            Eigen::Vector4d dT2;
            dT2 = 1.0/deltaT * B * dT1;

            Eigen::Vector4d ddT1(0.0,0.0,2.0,6.0*u);
            Eigen::Vector4d ddT2;
            ddT2 = 1.0/(deltaT*deltaT) * B * ddT1;

            vector<Eigen::Matrix4d> A,dA,ddA;
            A.resize(4);
            dA.resize(4);
            ddA.resize(4);
            for (int j=1;j<=3;j++)  // 0 to 2 ? 1 to 3 ?  :  j= 1 to 3, diff with <Spline-fusion>
            {
                Eigen::VectorXd upsilon_omega = Sophus::SE3d::log(SE3_vec[i+j-2].inverse() * SE3_vec[i+j-1]);
                Eigen::Matrix4d omega_mat;  // 4x4 se(3)
                omega_mat.setZero();
                omega_mat.block<3,3>(0,0) = skew<double>(upsilon_omega.tail<3>());
                omega_mat.block<3,1>(0,3) = upsilon_omega.head<3>();

                // calc A
                double B_select = T2(j);
                // \omega 4x4 = /omega 6x1
                //  [ w^ v]        [ v ]
                //  [ 0  0]        [ w ]
                //
                // while multiply a scalar, the same. (Ignore the last .at(3,3) 1)
                A[j] = (Sophus::SE3d::exp(B_select * upsilon_omega)).matrix();

                // calc dA
                double dB_select = dT2(j);
                dA[j] = A[j] * omega_mat * dB_select;

                // calc ddA
                double ddB_select = ddT2(j);
                ddA[j] = dA[j] * omega_mat * dB_select + A[j] * omega_mat * ddB_select;
            }

            Eigen::Matrix4d all;

            // get B-spline's R,T
            all = A[1] * A[2] * A[3];
            Eigen::Matrix4d ret = RTl0.matrix() * all;

            Eigen::Vector3d T,theta;
            Eigen::Matrix3d R;
            T = ret.block<3,1>(0,3);
            R = ret.block<3,3>(0,0);
            theta = R_to_ypr(R);
            fprintf(file,"%lf %lf %lf %lf %lf %lf %lf\n",
                          ts,
                          T(0),T(1),T(2),                                           
                          theta(0),theta(1),theta(2)
                   );

            // get B-spline's omega
            Eigen::Matrix4d dSE;
            all = dA[1]*A[2]*A[3] + A[1]*dA[2]*A[3] + A[1]*A[2]*dA[3];

            dSE = RTl0.matrix() * all;

            Eigen::Matrix3d skew_R = R.transpose() * dSE.block<3,3>(0,0);
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

            Eigen::Vector3d spline_acc = R.transpose() * (ddSE.block<3,1>(0,3) + Eigen::Vector3d(0,0,9.805));  // ? gravity not accurate

            fprintf(acc_file,"%lf %lf %lf %lf\n",
                              ts,spline_acc(0),spline_acc(1),spline_acc(2)
                   );
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc,argv,"cumulative_cubic_B_spline");
    ros::NodeHandle nh("~");

    ros::Publisher pub_path = nh.advertise<nav_msgs::Path>("path",1000);
    ros::Publisher pub_img = nh.advertise<sensor_msgs::Image>("image_grey",1000);  // cur frame
    ros::Publisher pub_img_info = nh.advertise<sensor_msgs::CameraInfo>("camera_info",1000);  // cur frame's caminfo
    ros::Publisher pub_depth = nh.advertise<sensor_msgs::Image>("depth",1000);  // cur depth
    ros::Publisher pub_depth_info = nh.advertise<sensor_msgs::CameraInfo>("depth_info",1000);  // cur frame's caminfo
    ros::Publisher pub_pc2 = nh.advertise<sensor_msgs::PointCloud2>("pc2",1000);  // cur frame's point cloud
    ros::Publisher pub_pixel_with_depth = nh.advertise<sensor_msgs::Image>("pixel_with_depth",1000);  // key frame with depth (green pixel)
    ros::Publisher pub_keyframe = nh.advertise<sensor_msgs::Image>("key_frame",1000);  // key frame
    ros::Publisher pub_estimated_img = nh.advertise<sensor_msgs::Image>("est_img",1000);  // est cur frame

    init();

    // ---- sync subscribe ----
    ros::Subscriber sub_imu = nh.subscribe("/imu0",1000,imu_callback);  // 200HZ (5ms)
    message_filters::Subscriber<sensor_msgs::Image> sub_img_l(nh,"/left/image_rect",1000);  // 20HZ (50ms)
    message_filters::Subscriber<stereo_msgs::DisparityImage> sub_disp(nh,"/disparity",1000);
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, stereo_msgs::DisparityImage> ExactPolicy;
    message_filters::Synchronizer<ExactPolicy> sync(ExactPolicy(1000), sub_img_l, sub_disp);
    sync.registerCallback(boost::bind(&vision_callback, _1, _2));
    // ------------------------

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

    ceres_process();

    mtx.unlock();
    ros::shutdown();

    return 0;
}
