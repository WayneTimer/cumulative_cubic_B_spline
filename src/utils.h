#ifndef utils_h
#define utils_h

#include <eigen3/Eigen/Dense>

Eigen::Vector3d R_to_ypr(const Eigen::Matrix3d& R);
Eigen::Matrix3d ypr_to_R(const Eigen::Vector3d& theta);
Eigen::Matrix3d skew(Eigen::Vector3d& A);

#endif
