#ifndef utils_h
#define utils_h

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

int double_equ_check(double x,double y,double eps);
Eigen::Vector3d R_to_ypr(const Eigen::Matrix3d& R);
Eigen::Matrix3d ypr_to_R(const Eigen::Vector3d& theta);
int cal_depth_img(cv::Mat& disparity,Eigen::MatrixXd& depth,double baseline,double f);

template <typename T>
Eigen::Matrix<T,3,3> skew(const Eigen::Matrix<T,3,1>& A)
{
    Eigen::Matrix<T,3,3> ret;

    ret(0,0) = T(0.0);
    ret(0,1) = -A(2);
    ret(0,2) = A(1);

    ret(1,0) = A(2);
    ret(1,1) = T(0.0);
    ret(1,2) = -A(0);

    ret(2,0) = -A(1);
    ret(2,1) = A(0);
    ret(2,2) = T(0.0);

    return ret;
}

#endif
