#include "utils.h"

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
