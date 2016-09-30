#include <cstdio>
#include <ceres/ceres.h>
#include <iostream>
#include <vector>
#include <sophus/se3.hpp>

#include "ceres_extensions.h"
#include "utils.h"
#include "ceres_solve.h"

using namespace std;

#define ACC_WEIGHT 0.05
#define OMEGA_WEIGHT 0.2

extern vector<Sophus::SE3d> SE3_vec;
extern vector<double> pose_ts_vec;
extern vector<double> imu_ts_vec;
extern vector<Eigen::Vector3d> imu_acc_vec;
extern vector<Eigen::Vector3d> imu_omega_vec;
extern Eigen::Matrix4d B;
extern double deltaT;

vector<Sophus::SE3d> new_SE3_vec;
int imu_idx;
double acc_bias[3];
double omega_bias[3];
vector<Eigen::Vector3d> acc_bias_vec;
vector<Eigen::Vector3d> omega_bias_vec;

struct vio_functor
{
    // first (p,q) -> to be optimized
    // last (p,q) -> constant (initial guess)
    // residual: 7x1
    template <typename T>
    bool operator() (const T* const p, const T* const q,
                     const T* const p0, const T* const q0,
                     T* residual) const
    {
        for (int i=0;i<3;i++)
            residual[i] = p[i] - p0[i];
        for (int i=0;i<4;i++)
            residual[3+i] = q[i] - q0[i];
        return true;
    }
};

struct acc_functor
{
    template <typename T>
    bool operator() (const T* const p0, const T* const q0,
                     const T* const p1, const T* const q1,
                     const T* const p2, const T* const q2,
                     const T* const p3, const T* const q3,
                     const T* const a_bias,

                     const T* const imu_info,

                     T* residual) const
    {
        vector<Eigen::Matrix<T,4,4> > A,dA,ddA;
        A.resize(4);
        dA.resize(4);
        ddA.resize(4);

        vector<Sophus::SE3Group<T> > omega_SE3;
        omega_SE3.resize(4);

        Eigen::Matrix<T,3,1> translation;
        Eigen::Quaternion<T> quat;  // Eigen::Quaternion (w,x,y,z)

        translation[0] = p0[0];
        translation[1] = p0[1];
        translation[2] = p0[2];
        quat = Eigen::Quaternion<T>(q0[3],q0[0],q0[1],q0[2]);
        omega_SE3[0] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p1[0];
        translation[1] = p1[1];
        translation[2] = p1[2];
        quat = Eigen::Quaternion<T>(q1[3],q1[0],q1[1],q1[2]);
        omega_SE3[1] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p2[0];
        translation[1] = p2[1];
        translation[2] = p2[2];
        quat = Eigen::Quaternion<T>(q2[3],q2[0],q2[1],q2[2]);
        omega_SE3[2] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p3[0];
        translation[1] = p3[1];
        translation[2] = p3[2];
        quat = Eigen::Quaternion<T>(q3[3],q3[0],q3[1],q3[2]);
        omega_SE3[3] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);
        // ===============================

        Sophus::SE3Group<T> RTl0 = omega_SE3[0];

        // ---- construct B ----
        Eigen::Matrix<T,4,4> B;
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

        Eigen::Matrix<T,4,4> tmp_B;
        tmp_B = T(1.0/6.0) * B;
        B = tmp_B;
        // --------------------

        Eigen::Matrix<T,4,1> T1( T(1.0), imu_info[0], imu_info[0]*imu_info[0], imu_info[0]*imu_info[0]*imu_info[0]);
        Eigen::Matrix<T,4,1> T2;
        T2 = B*T1;

        Eigen::Matrix<T,4,1> dT1( T(0.0), T(1.0), T(2.0)*imu_info[0], T(3.0)*imu_info[0]*imu_info[0]);
        Eigen::Matrix<T,4,1> dT2;
        dT2 = T(1.0/deltaT) * B * dT1;

        Eigen::Matrix<T,4,1> ddT1( T(0.0), T(0.0), T(2.0), T(6.0)*imu_info[0]);
        Eigen::Matrix<T,4,1> ddT2;
        ddT2 = T(1.0/(deltaT*deltaT)) * B * ddT1;

        for (int j=1;j<=3;j++)  // 0 to 2 ? 1 to 3 ?  :  j= 1 to 3, diff with <Spline-fusion>
        {
            Eigen::Matrix<T,6,1> upsilon_omega = Sophus::SE3Group<T>::log(omega_SE3[j-1].inverse() * omega_SE3[j]);
            Eigen::Matrix<T,4,4> omega_mat;  // 4x4 se(3)
            omega_mat.setZero();
            Eigen::Matrix<T,3,1> for_skew;
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

        Eigen::Matrix<T,4,4> all;

        // get B-spline's R,T
        all = A[1] * A[2] * A[3];
        Eigen::Matrix<T,4,4> ret = RTl0.matrix() * all;

        Eigen::Matrix<T,3,3> R;
        R = ret.template block<3,3>(0,0);

        // get B-spline's acc
        Eigen::Matrix<T,4,4> ddSE;
        all =   ddA[1]*A[2]*A[3] + A[1]*ddA[2]*A[3] + A[1]*A[2]*ddA[3]
              + T(2.0)*dA[1]*dA[2]*A[3] + T(2.0)*dA[1]*A[2]*dA[3] + T(2.0)*A[1]*dA[2]*dA[3];
        ddSE = RTl0.matrix() * all;

        Eigen::Matrix<T,3,1> spline_acc = R.transpose() * (ddSE.template block<3,1>(0,3) + Eigen::Matrix<T,3,1>(T(0),T(0),T(9.805)));  // ? gravity not accurate


        // ---- get residual ----
        residual[0] = (spline_acc(0,0) + a_bias[0] - imu_info[1]) * T(ACC_WEIGHT);
        residual[1] = (spline_acc(1,0) + a_bias[1] - imu_info[2]) * T(ACC_WEIGHT);
        residual[2] = (spline_acc(2,0) + a_bias[2] - imu_info[3]) * T(ACC_WEIGHT);

        return true;
    }
};

struct omega_functor
{
    template <typename T>
    bool operator() (const T* const p0, const T* const q0,
                     const T* const p1, const T* const q1,
                     const T* const p2, const T* const q2,
                     const T* const p3, const T* const q3,
                     const T* const w_bias,

                     const T* const imu_info,

                     T* residual) const
    {
        vector<Eigen::Matrix<T,4,4> > A,dA;
        A.resize(4);
        dA.resize(4);

        // ========  construct SE3 ===========
        vector<Sophus::SE3Group<T> > omega_SE3;
        omega_SE3.resize(4);

        Eigen::Matrix<T,3,1> translation;
        Eigen::Quaternion<T> quat;  // Eigen::Quaternion (w,x,y,z)

        translation[0] = p0[0];
        translation[1] = p0[1];
        translation[2] = p0[2];
        quat = Eigen::Quaternion<T>(q0[3],q0[0],q0[1],q0[2]);
        omega_SE3[0] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p1[0];
        translation[1] = p1[1];
        translation[2] = p1[2];
        quat = Eigen::Quaternion<T>(q1[3],q1[0],q1[1],q1[2]);
        omega_SE3[1] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p2[0];
        translation[1] = p2[1];
        translation[2] = p2[2];
        quat = Eigen::Quaternion<T>(q2[3],q2[0],q2[1],q2[2]);
        omega_SE3[2] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p3[0];
        translation[1] = p3[1];
        translation[2] = p3[2];
        quat = Eigen::Quaternion<T>(q3[3],q3[0],q3[1],q3[2]);
        omega_SE3[3] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);
        // ===============================

        Sophus::SE3Group<T> RTl0 = omega_SE3[0];

        Eigen::Matrix<T,4,1> T1( T(1.0), imu_info[0], imu_info[0]*imu_info[0], imu_info[0]*imu_info[0]*imu_info[0]);
        Eigen::Matrix<T,4,1> T2;
        T2 = B*T1;

        Eigen::Matrix<T,4,1> dT1( T(0.0), T(1.0), T(2.0)*imu_info[0], T(3.0)*imu_info[0]*imu_info[0]);
        Eigen::Matrix<T,4,1> dT2;
        dT2 = T(1.0/deltaT) * B * dT1;

        for (int j=1;j<=3;j++)  // 0 to 2 ? 1 to 3 ?  :  j= 1 to 3, diff with <Spline-fusion>
        {
            Eigen::Matrix<T,6,1> upsilon_omega = Sophus::SE3Group<T>::log(omega_SE3[j-1].inverse() * omega_SE3[j]);
            Eigen::Matrix<T,4,4> omega_mat;  // 4x4 se(3)
            omega_mat.setZero();
            Eigen::Matrix<T,3,1> for_skew;
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
        }

        Eigen::Matrix<T,4,4> all;

        // get B-spline's R,T
        all = A[1] * A[2] * A[3];
        Eigen::Matrix<T,4,4> ret = RTl0.matrix() * all;

        Eigen::Matrix<T,3,3> R;
        R = ret.template block<3,3>(0,0);

        // get B-spline's omega
        Eigen::Matrix<T,4,4> dSE;
        all = dA[1]*A[2]*A[3] + A[1]*dA[2]*A[3] + A[1]*A[2]*dA[3];

        dSE = RTl0.matrix() * all;

        Eigen::Matrix<T,3,3> skew_R;
        skew_R = R.transpose() * dSE.template block<3,3>(0,0);
        T wx,wy,wz;  // ? simple mean
        wx = (-skew_R(1,2) + skew_R(2,1)) / T(2.0);
        wy = (-skew_R(2,0) + skew_R(0,2)) / T(2.0);
        wz = (-skew_R(0,1) + skew_R(1,0)) / T(2.0);


        // ---- get residual ----
        residual[0] = (wx + w_bias[0] - imu_info[4]) * T(OMEGA_WEIGHT);
        residual[1] = (wy + w_bias[1] - imu_info[5]) * T(OMEGA_WEIGHT);
        residual[2] = (wz + w_bias[2] - imu_info[6]) * T(OMEGA_WEIGHT);

        return true;
    }
};

struct blur_vo_functor
{
    template <typename T>
    bool operator() (const T* const p0, const T* const q0,
                     const T* const p1, const T* const q1,
                     const T* const p2, const T* const q2,
                     const T* const p3, const T* const q3,

                     const T* const imu_info,

                     T* residual) const
    {
        return true;
    }
};

// when get 4 state, solve, update, slid
// (head), (head+1), (head+2), (head+3)
void ceres_solve(int head)
{
    ceres::Problem problem;
    ceres::CostFunction* cost_function;

    double p[4][3];
    double q[4][4];
    double p0[4][3];
    double q0[4][4];
    ceres::LocalParameterization *local_parameterization = new ceres_ext::EigenQuaternionParameterization();

    // add vio constraint
    for (int i=0;i<4;i++)
    {
        Eigen::Vector3d T = new_SE3_vec[head+i].translation();
        Eigen::Quaterniond quat(new_SE3_vec[head+i].rotationMatrix());

        for (int j=0;j<3;j++)
        {
            p0[i][j] = T[j];
            p[i][j] = T[j];
        }
        q0[i][0] = quat.x(), q0[i][1] = quat.y(), q0[i][2] = quat.z(), q0[i][3] = quat.w();  // q = {x,y,z,w}
        q[i][0] = quat.x(), q[i][1] = quat.y(), q[i][2] = quat.z(), q[i][3] = quat.w();

        cost_function = new ceres::AutoDiffCostFunction<vio_functor, 7, 3,4,3,4>(new vio_functor);
        problem.AddResidualBlock(cost_function,NULL,&p[i][0],&q[i][0],&p0[i][0],&q0[i][0]);

        problem.AddParameterBlock(&q[i][0],4,local_parameterization);  // q = {x,y,z,w}
        problem.SetParameterBlockConstant(&p0[i][0]);
        problem.SetParameterBlockConstant(&q0[i][0]);
    }

    // add acc & omega constraint   ----  Attention! Probably needs a less weight
    // t \in [ head+1, head+2 )
    double ts_down_limit,ts_up_limit;
    ts_down_limit = pose_ts_vec[head+1];
    ts_up_limit = pose_ts_vec[head+2];
    int l = imu_ts_vec.size();

    int imu_cnt = 0;
    vector<double*> imu_info_list;
    imu_info_list.clear();

    for (;imu_idx<l;imu_idx++)
    {
        if (ts_down_limit > imu_ts_vec[imu_idx]) continue;
        if (imu_ts_vec[imu_idx] >= ts_up_limit) break;

        imu_info_list.push_back( (double*)malloc(sizeof(double)*7) );  // malloc 1+3+3 double  (u,acc,omega)
        *(imu_info_list[imu_cnt]) = (imu_ts_vec[imu_idx]-pose_ts_vec[head+1])/deltaT;
        *(imu_info_list[imu_cnt]+1) = imu_acc_vec[imu_idx][0];
        *(imu_info_list[imu_cnt]+2) = imu_acc_vec[imu_idx][1];
        *(imu_info_list[imu_cnt]+3) = imu_acc_vec[imu_idx][2];
        *(imu_info_list[imu_cnt]+4) = imu_omega_vec[imu_idx][0];
        *(imu_info_list[imu_cnt]+5) = imu_omega_vec[imu_idx][1];
        *(imu_info_list[imu_cnt]+6) = imu_omega_vec[imu_idx][2];

        cost_function = new ceres::AutoDiffCostFunction<acc_functor, 3, 3,4,3,4,3,4,3,4, 3, 7>(new acc_functor);
        problem.AddResidualBlock(cost_function,NULL, &p[0][0],&q[0][0],&p[1][0],&q[1][0],&p[2][0],&q[2][0],&p[3][0],&q[3][0],
                                                     acc_bias,
                                                     imu_info_list[imu_cnt]
                                );

        cost_function = new ceres::AutoDiffCostFunction<omega_functor, 3, 3,4,3,4,3,4,3,4, 3, 7>(new omega_functor);
        problem.AddResidualBlock(cost_function,NULL, &p[0][0],&q[0][0],&p[1][0],&q[1][0],&p[2][0],&q[2][0],&p[3][0],&q[3][0],
                                                     omega_bias,
                                                     imu_info_list[imu_cnt]
                                );

        problem.SetParameterBlockConstant(imu_info_list[imu_cnt]);

        imu_cnt++;
    }

    // solving
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.line_search_sufficient_function_decrease = 1e-3;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;

    // free constant parameters
    for (int i=0;i<imu_cnt;i++)
    {
        free(imu_info_list[i]);
    }

    // update
    for (int i=0;i<4;i++)
    {
        Eigen::Vector3d T;
        Eigen::Quaterniond quat;
        for (int j=0;j<3;j++)
            T[j] = p[i][j];
        quat.x() = q[i][0];
        quat.y() = q[i][1];
        quat.z() = q[i][2];
        quat.w() = q[i][3];

        new_SE3_vec[head+i] = Sophus::SE3d(quat.toRotationMatrix(),T);
    }
}

void output_result()
{
    FILE *solve_file;  // Solved B-spline: ts p \theta
    FILE *solve_omega_file;  // Solved B-spline': ts \omega
    FILE *solve_vel_file;  // Solved B-spline': ts vel
    FILE *solve_acc_file;  // Solved B-spline'': ts acc
    FILE *bias_file;  // ts acc_bias omega_bias

    solve_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/solve_pose.txt","w");
    solve_omega_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/solve_omega.txt","w");
    solve_vel_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/solve_vel.txt","w");
    solve_acc_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/solve_acc.txt","w");
    bias_file = fopen("/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/bias.txt","w");

    //---------------------
    double diff = 0.001;  // 1ms intepolate
    int l;
    l = new_SE3_vec.size();
    for (int i=1;i<l-2;i++)
    {
        Sophus::SE3d RTl0 = new_SE3_vec[i-1];
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
                Eigen::VectorXd upsilon_omega = Sophus::SE3d::log(new_SE3_vec[i+j-2].inverse() * new_SE3_vec[i+j-1]);
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
            fprintf(solve_file,"%lf %lf %lf %lf %lf %lf %lf\n",
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

            fprintf(solve_omega_file,"%lf %lf %lf %lf\n",
                                      ts,
                                      wx,wy,wz
                   );
            fprintf(solve_vel_file,"%lf %lf %lf %lf\n",
                                    ts,
                                    linear_vel(0),linear_vel(1),linear_vel(2)
                   );

            // get B-spline's acc
            Eigen::Matrix4d ddSE;
            all =   ddA[1]*A[2]*A[3] + A[1]*ddA[2]*A[3] + A[1]*A[2]*ddA[3]
                  + 2.0*dA[1]*dA[2]*A[3] + 2.0*dA[1]*A[2]*dA[3] + 2.0*A[1]*dA[2]*dA[3];
            ddSE = RTl0.matrix() * all;

            Eigen::Vector3d spline_acc = R.transpose() * (ddSE.block<3,1>(0,3) + Eigen::Vector3d(0,0,9.805));  // ? gravity not accurate

            fprintf(solve_acc_file,"%lf %lf %lf %lf\n",
                                    ts,spline_acc(0),spline_acc(1),spline_acc(2)
                   );
        }
    }

    // output bias
    for (int i=0;i<l-3;i++)
    {
        fprintf(bias_file,"%lf %lf %lf %lf %lf %lf %lf\n",
                           pose_ts_vec[i],
                           acc_bias_vec[i][0],acc_bias_vec[i][1],acc_bias_vec[i][2],
                           omega_bias_vec[i][0],omega_bias_vec[i][1],omega_bias_vec[i][2]
               );
    }

    fclose(solve_file);
    fclose(solve_omega_file);
    fclose(solve_vel_file);
    fclose(solve_acc_file);
    fclose(bias_file);
}

void ceres_process()
{
    new_SE3_vec.clear();
    imu_idx = 0;
    int l;
    l = SE3_vec.size();
    new_SE3_vec.resize(l);
    for (int i=0;i<l;i++)
        new_SE3_vec[i] = SE3_vec[i];

    // set bias initial guess
    for (int i=0;i<3;i++)
    {
        acc_bias[i] = 0.0;
        omega_bias[i] = 0.0;
    }
    acc_bias_vec.clear();
    omega_bias_vec.clear();

    for (int i=0;i<l-3;i++)
    {
        ceres_solve(i);
        acc_bias_vec.push_back(Eigen::Vector3d(acc_bias[0],acc_bias[1],acc_bias[2]));
        omega_bias_vec.push_back(Eigen::Vector3d(omega_bias[0],omega_bias[1],omega_bias[2]));
    }

    // ---- output to file ----
    output_result();
}
