#include "utils.h"
#include <cv_bridge/cv_bridge.h>

// 0:equ, -1:left<right, 1:left>right
int double_equ_check(double x,double y,double eps)
{
    double t;
    t = x-y;
    if (t<-eps) return -1;
    if (t>eps) return 1;
    return 0;
}

// Shen's 6910P L2.pdf (P20): Z_1 Y_2 X_3
Eigen::Vector3d R_to_ypr(const Eigen::Matrix3d& R)
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0)*cos(y)+n(1)*sin(y));
    double r = atan2(a(0)*sin(y)-a(1)*cos(y), -o(0)*sin(y)+o(1)*cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr;
}

Eigen::Matrix3d ypr_to_R(const Eigen::Vector3d& theta)
{
    double e1,e2,e3;
    e1 = theta(0);
    e2 = theta(1);
    e3 = theta(2);

    Eigen::Matrix3d R;

    R(0,0) = cos(e1)*cos(e2);
    R(0,1) = cos(e1)*sin(e2)*sin(e3) - cos(e3)*sin(e1);
    R(0,2) = sin(e1)*sin(e3) + cos(e1)*cos(e3)*sin(e2);

    R(1,0) = cos(e2)*sin(e1);
    R(1,1) = cos(e1)*cos(e3) + sin(e1)*sin(e2)*sin(e3);
    R(1,2) = cos(e3)*sin(e1)*sin(e2) - cos(e1)*sin(e3);

    R(2,0) = -sin(e2);
    R(2,1) = cos(e2)*sin(e3);
    R(2,2) = cos(e2)*cos(e3);

    return R;
}

// get depth from disparity
int cal_depth_img(cv::Mat& disparity,Eigen::MatrixXd& depth,double baseline,double f)
{
    int depth_cnt = 0;
    int h = disparity.rows;
    int w = disparity.cols;

    depth = Eigen::MatrixXd::Zero(h,w);

    for (int u=0;u<w;u++)
        for (int v=0;v<h;v++)
        {
            float d = disparity.at<float>(v,u);
            if ( d<2.0 )  // default: 3.0
                depth(v,u) = 0.0;
            else
            {
                depth(v,u) = baseline * f / d;
                depth_cnt++;
            }
        }
    return depth_cnt;
}

sensor_msgs::Image img2msg(cv::Mat& img, ros::Time& ros_stamp, string encoding)
{
    cv_bridge::CvImage cvimg;
    cvimg.header.stamp = ros_stamp;
    cvimg.encoding = encoding;
    cvimg.image = img;
    return *cvimg.toImageMsg();
}
