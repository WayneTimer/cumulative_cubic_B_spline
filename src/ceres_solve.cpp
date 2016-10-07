//#define UNBLUR

#include <cstdio>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <iostream>
#include <vector>
#include <sophus/se3.hpp>

#include "ceres_extensions.h"
#include "utils.h"
#include "ceres_solve.h"
#include "data_structure.h"

using namespace std;

#define DVO_WEIGHT 1.0

#define BLUR_WEIGHT 1.0
#define PRIOR_WEIGHT 0.001
#define ACC_WEIGHT 0.2
#define OMEGA_WEIGHT 1.0
#define INTERPOLATE_DIFF 0.01

extern Graph graph;
extern CALI_PARA cali;
extern double deltaT;
extern int calc_level;
extern FILE *solve_file;  // Solved B-spline: ts p \theta
extern FILE *solve_omega_file;  // Solved B-spline': ts \omega
extern FILE *solve_vel_file;  // Solved B-spline': ts vel
extern FILE *solve_acc_file;  // Solved B-spline'': ts acc
extern int calc_level;
extern int key_frame_no;

/*
B-Spline's SE(3) = (R_i^0,T_i^0) = SE(3)_i^0
Have SE(3)_i^0, SE(3)_j^0
SE(3)_i^j = ( SE(3)_j^0 )^-1 * SE(3)_i^0
SE(3)_i^j * [x,y,z,1]^i = [x,y,z,1]^j   (transform a point from key-frame i to cur-frame j)
*/
vector<Sophus::SE3d> SE3_vec;
int imu_idx;

// (head), (head+1), (head+2), (head+3)
//        key_frame  ( blur ]
class blur_vo_functor  // the whole image residual
{
private:
    const ceres::BiCubicInterpolator< ceres::Grid2D<double,1> >& img;
    int height,width;
    double fx,fy,cx,cy;
    double exposure_time_u;  // in u domain
    const Eigen::MatrixXd& key_frame_img;
    const Eigen::MatrixXd& key_frame_depth;
    int ref_u,ref_v; // v - height, u - width

public:
    blur_vo_functor(
                    const Eigen::MatrixXd& _key_frame_img,
                    const Eigen::MatrixXd& _key_frame_depth,
                    const ceres::BiCubicInterpolator< ceres::Grid2D<double,1> >& _img,
                    int _height, int _width,
                    double _fx, double _fy, double _cx, double _cy, double _exposure_time_u,
                    int _ref_u, int _ref_v
                   ) :
                    key_frame_img(_key_frame_img),
                    key_frame_depth(_key_frame_depth),
                    img(_img),
                    height(_height), width(_width),
                    fx(_fx), fy(_fy), cx(_cx), cy(_cy), exposure_time_u(_exposure_time_u),
                    ref_u(_ref_u), ref_v(_ref_v)
    {}

    template <typename T>
    bool operator() (const T* const p0, const T* const q0,
                     const T* const p1, const T* const q1,
                     const T* const p2, const T* const q2,
                     const T* const p3, const T* const q3,

                     // no need to give img stamp - ts(already delayed), and exposure period [ts-EXP, ts), in u
                     // is always 1.0
                     const T* const key_frame_RT,

                     T* residual) const
    {
        vector<Eigen::Matrix<T,4,4> > A,dA;
        A.resize(4);
        dA.resize(4);

        // ========  construct SE3 ===========
        vector<Sophus::SE3Group<T> > SE3;
        SE3.resize(4);

        Eigen::Matrix<T,3,1> translation;
        Eigen::Quaternion<T> quat;  // Eigen::Quaternion (w,x,y,z)

        translation[0] = p0[0];
        translation[1] = p0[1];
        translation[2] = p0[2];
        quat = Eigen::Quaternion<T>(q0[3],q0[0],q0[1],q0[2]);
        SE3[0] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p1[0];
        translation[1] = p1[1];
        translation[2] = p1[2];
        quat = Eigen::Quaternion<T>(q1[3],q1[0],q1[1],q1[2]);
        SE3[1] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p2[0];
        translation[1] = p2[1];
        translation[2] = p2[2];
        quat = Eigen::Quaternion<T>(q2[3],q2[0],q2[1],q2[2]);
        SE3[2] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p3[0];
        translation[1] = p3[1];
        translation[2] = p3[2];
        quat = Eigen::Quaternion<T>(q3[3],q3[0],q3[1],q3[2]);
        SE3[3] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);
        // ===============================

        Sophus::SE3Group<T> RTl0 = SE3[0];

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

        // ---- samples to generate blur ----
        vector< Sophus::SE3Group<T> > SE3_j;
        SE3_j.clear();
        for ( T tu=T(1.0-exposure_time_u);tu<T(1.0);tu=tu+T(INTERPOLATE_DIFF) )
        {
            Eigen::Matrix<T,4,1> T1( T(1.0), tu, tu*tu, tu*tu*tu);
            Eigen::Matrix<T,4,1> T2;
            T2 = B*T1;

            Eigen::Matrix<T,4,1> dT1( T(0.0), T(1.0), T(2.0)*tu, T(3.0)*tu*tu);
            Eigen::Matrix<T,4,1> dT2;
            dT2 = T(1.0/deltaT) * B * dT1;

            for (int j=1;j<=3;j++)  // 0 to 2 ? 1 to 3 ?  :  j= 1 to 3, diff with <Spline-fusion>
            {
                Eigen::Matrix<T,6,1> upsilon_omega = Sophus::SE3Group<T>::log(SE3[j-1].inverse() * SE3[j]);
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
            Eigen::Matrix<T,3,1> trans;
            trans = ret.template block<3,1>(0,3);

            SE3_j.push_back( Sophus::SE3Group<T>(R,trans) );
        }

        // ===== generate blur img =====
        residual[0] = T(0.0);

        // --- get key-frame RT ---
        Sophus::SE3Group<T> key_frame_SE3;  // (R,T)_i^0
        translation[0] = key_frame_RT[0];
        translation[1] = key_frame_RT[1];
        translation[2] = key_frame_RT[2];
        quat = Eigen::Quaternion<T>(key_frame_RT[6],key_frame_RT[3],key_frame_RT[4],key_frame_RT[5]);
        key_frame_SE3 = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);
        // -------

        int blur_inter_cnt = SE3_j.size();
        int v = ref_v;
        int u = ref_u;

        T resi = T(0.0);
        double te_cnt = 0.0;
        for (int j=0;j<blur_inter_cnt;j++)
        {
            // SE(3)_i^j = ( SE(3)_j^0 )^-1 * SE(3)_i^0
            Sophus::SE3Group<T> SE3_i_2_j = SE3_j[j].inverse() * key_frame_SE3;

            Eigen::Matrix<T,3,1> p_ref,p_cur;  // [x,y,z]
            double lambda;
            lambda = key_frame_depth(v,u);

            if (double_equ_check(lambda,0.0,DOUBLE_EPS)<=0) // no depth
                continue;

            p_ref[0] = T( (u-cx)/fx * lambda );
            p_ref[1] = T( (v-cy)/fy * lambda );
            p_ref[2] = T( lambda );

            p_cur = SE3_i_2_j * p_ref;

            T u_new,v_new;
            u_new = (p_cur[0]/p_cur[2]) * T(fx) + T(cx);
            v_new = (p_cur[1]/p_cur[2]) * T(fy) + T(cy);

            if (u_new < T(1) || u_new >= T(width-1) || v_new < T(1) || v_new >= T(height-1) )
                continue;

            T inten;
            img.Evaluate(v_new,u_new,&inten);

            double inten_est;
            inten_est = key_frame_img(v,u);

            T tmp_resi = T(inten_est) - inten;

            resi = resi + tmp_resi;
            te_cnt = te_cnt + 1.0;
        }
        if (double_equ_check(te_cnt,0.0,DOUBLE_EPS)>0) // ensure te_cnt > 0.0
        {
            resi = resi / T(te_cnt);
        }

        T resi_weight = T(1.0);
        if ( resi > T(5.0) )
        {
            resi_weight = T(5.0) / resi;
        }
        if ( resi < T(-5.0) )
        {
            resi_weight = T(-5.0) / resi;
        }

        residual[0] = resi * resi_weight * T(BLUR_WEIGHT);

        return true;
    }
};

// (head), (head+1), (head+2), (head+3)
//        key_frame  ( blur ]
class dvo_functor  // the whole image residual
{
private:
    const ceres::BiCubicInterpolator< ceres::Grid2D<double,1> >& img;
    int height,width;
    double fx,fy,cx,cy;
    double exposure_time_u;  // in u domain
    const Eigen::MatrixXd& key_frame_img;
    const Eigen::MatrixXd& key_frame_depth;
    int ref_u,ref_v; // v - height, u - width

public:
    dvo_functor(
                    const Eigen::MatrixXd& _key_frame_img,
                    const Eigen::MatrixXd& _key_frame_depth,
                    const ceres::BiCubicInterpolator< ceres::Grid2D<double,1> >& _img,
                    int _height, int _width,
                    double _fx, double _fy, double _cx, double _cy, double _exposure_time_u,
                    int _ref_u, int _ref_v
               ) :
                    key_frame_img(_key_frame_img),
                    key_frame_depth(_key_frame_depth),
                    img(_img),
                    height(_height), width(_width),
                    fx(_fx), fy(_fy), cx(_cx), cy(_cy), exposure_time_u(_exposure_time_u),
                    ref_u(_ref_u), ref_v(_ref_v)
    {}

    template <typename T>
    bool operator() (const T* const p0, const T* const q0,
                     const T* const p1, const T* const q1,
                     const T* const p2, const T* const q2,
                     const T* const p3, const T* const q3,

                     // no need to give img stamp - ts(already delayed), and exposure period [ts-EXP, ts), in u
                     // is always 1.0
                     const T* const key_frame_RT,

                     T* residual) const
    {
        // ========  construct SE3 ===========
        vector<Sophus::SE3Group<T> > SE3;
        SE3.resize(3);

        Eigen::Matrix<T,3,1> translation;
        Eigen::Quaternion<T> quat;  // Eigen::Quaternion (w,x,y,z)

        translation[0] = p0[0];
        translation[1] = p0[1];
        translation[2] = p0[2];
        quat = Eigen::Quaternion<T>(q0[3],q0[0],q0[1],q0[2]);
        SE3[0] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p1[0];
        translation[1] = p1[1];
        translation[2] = p1[2];
        quat = Eigen::Quaternion<T>(q1[3],q1[0],q1[1],q1[2]);
        SE3[1] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p2[0];
        translation[1] = p2[1];
        translation[2] = p2[2];
        quat = Eigen::Quaternion<T>(q2[3],q2[0],q2[1],q2[2]);
        SE3[2] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);
        // ===============================

        // --- get key-frame RT ---
        Sophus::SE3Group<T> key_frame_SE3;  // (R,T)_i^0
        translation[0] = key_frame_RT[0];
        translation[1] = key_frame_RT[1];
        translation[2] = key_frame_RT[2];
        quat = Eigen::Quaternion<T>(key_frame_RT[6],key_frame_RT[3],key_frame_RT[4],key_frame_RT[5]);
        key_frame_SE3 = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);
        // -------

        // ===== generate dense img =====
        residual[0] = T(0.0);

        int v = ref_v;
        int u = ref_u;

        // SE(3)_i^j = ( SE(3)_j^0 )^-1 * SE(3)_i^0
        Sophus::SE3Group<T> SE3_i_2_j = SE3[2].inverse() * key_frame_SE3;

        Eigen::Matrix<T,3,1> p_ref,p_cur;  // [x,y,z]
        double lambda;
        lambda = key_frame_depth(v,u);

        if (double_equ_check(lambda,0.0,DOUBLE_EPS)<=0) // no depth
            return true;

        p_ref[0] = T( (u-cx)/fx * lambda );
        p_ref[1] = T( (v-cy)/fy * lambda );
        p_ref[2] = T( lambda );

        p_cur = SE3_i_2_j * p_ref;

        T u_new,v_new;
        u_new = (p_cur[0]/p_cur[2]) * T(fx) + T(cx);
        v_new = (p_cur[1]/p_cur[2]) * T(fy) + T(cy);

        if (u_new < T(1) || u_new >= T(width-1) || v_new < T(1) || v_new >= T(height-1) )
            return true;

        T inten;
        img.Evaluate(v_new,u_new,&inten);

        double inten_est;
        inten_est = key_frame_img(v,u);

        T resi = T(inten_est) - inten;

        T resi_weight = T(1.0);
        if ( resi > T(5.0) )
        {
            resi_weight = T(5.0) / resi;
        }
        if ( resi < T(-5.0) )
        {
            resi_weight = T(-5.0) / resi;
        }

        residual[0] = resi * resi_weight * T(DVO_WEIGHT);

        return true;
    }
};

struct prior_functor
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
            residual[i] = (p[i] - p0[i]) * T(PRIOR_WEIGHT);
        for (int i=0;i<4;i++)
            residual[3+i] = (q[i] - q0[i]) * T(PRIOR_WEIGHT);
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

                     const T* const imu_info,

                     T* residual) const
    {
        vector<Eigen::Matrix<T,4,4> > A,dA,ddA;
        A.resize(4);
        dA.resize(4);
        ddA.resize(4);

        vector<Sophus::SE3Group<T> > SE3;
        SE3.resize(4);

        Eigen::Matrix<T,3,1> translation;
        Eigen::Quaternion<T> quat;  // Eigen::Quaternion (w,x,y,z)

        translation[0] = p0[0];
        translation[1] = p0[1];
        translation[2] = p0[2];
        quat = Eigen::Quaternion<T>(q0[3],q0[0],q0[1],q0[2]);
        SE3[0] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p1[0];
        translation[1] = p1[1];
        translation[2] = p1[2];
        quat = Eigen::Quaternion<T>(q1[3],q1[0],q1[1],q1[2]);
        SE3[1] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p2[0];
        translation[1] = p2[1];
        translation[2] = p2[2];
        quat = Eigen::Quaternion<T>(q2[3],q2[0],q2[1],q2[2]);
        SE3[2] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p3[0];
        translation[1] = p3[1];
        translation[2] = p3[2];
        quat = Eigen::Quaternion<T>(q3[3],q3[0],q3[1],q3[2]);
        SE3[3] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);
        // ===============================

        Sophus::SE3Group<T> RTl0 = SE3[0];

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
            Eigen::Matrix<T,6,1> upsilon_omega = Sophus::SE3Group<T>::log(SE3[j-1].inverse() * SE3[j]);
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

        Eigen::Matrix<T,3,1> spline_acc = R.transpose() * ddSE.template block<3,1>(0,3);  // ? g0 has been removed

        // ---- get residual ----
        residual[0] = (spline_acc(0,0) - imu_info[1]) * T(ACC_WEIGHT);
        residual[1] = (spline_acc(1,0) - imu_info[2]) * T(ACC_WEIGHT);
        residual[2] = (spline_acc(2,0) - imu_info[3]) * T(ACC_WEIGHT);

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

                     const T* const imu_info,

                     T* residual) const
    {
        vector<Eigen::Matrix<T,4,4> > A,dA;
        A.resize(4);
        dA.resize(4);

        // ========  construct SE3 ===========
        vector<Sophus::SE3Group<T> > SE3;
        SE3.resize(4);

        Eigen::Matrix<T,3,1> translation;
        Eigen::Quaternion<T> quat;  // Eigen::Quaternion (w,x,y,z)

        translation[0] = p0[0];
        translation[1] = p0[1];
        translation[2] = p0[2];
        quat = Eigen::Quaternion<T>(q0[3],q0[0],q0[1],q0[2]);
        SE3[0] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p1[0];
        translation[1] = p1[1];
        translation[2] = p1[2];
        quat = Eigen::Quaternion<T>(q1[3],q1[0],q1[1],q1[2]);
        SE3[1] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p2[0];
        translation[1] = p2[1];
        translation[2] = p2[2];
        quat = Eigen::Quaternion<T>(q2[3],q2[0],q2[1],q2[2]);
        SE3[2] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);

        translation[0] = p3[0];
        translation[1] = p3[1];
        translation[2] = p3[2];
        quat = Eigen::Quaternion<T>(q3[3],q3[0],q3[1],q3[2]);
        SE3[3] = Sophus::SE3Group<T>(quat.toRotationMatrix(),translation);
        // ===============================

        Sophus::SE3Group<T> RTl0 = SE3[0];

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

        for (int j=1;j<=3;j++)  // 0 to 2 ? 1 to 3 ?  :  j= 1 to 3, diff with <Spline-fusion>
        {
            Eigen::Matrix<T,6,1> upsilon_omega = Sophus::SE3Group<T>::log(SE3[j-1].inverse() * SE3[j]);
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
        residual[0] = (wx - imu_info[4]) * T(OMEGA_WEIGHT);
        residual[1] = (wy - imu_info[5]) * T(OMEGA_WEIGHT);
        residual[2] = (wz - imu_info[6]) * T(OMEGA_WEIGHT);

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

    // prepare prior constraint
    for (int i=0;i<4;i++)
    {
        Eigen::Vector3d trans = SE3_vec[i].translation();
        Eigen::Quaterniond quat(SE3_vec[i].rotationMatrix());

        for (int j=0;j<3;j++)
        {
            p0[i][j] = trans[j];
            p[i][j] = trans[j];
        }
        q0[i][0] = quat.x(), q0[i][1] = quat.y(), q0[i][2] = quat.z(), q0[i][3] = quat.w();  // q = {x,y,z,w}
        q[i][0] = quat.x(), q[i][1] = quat.y(), q[i][2] = quat.z(), q[i][3] = quat.w();

        problem.AddParameterBlock(&q[i][0],4,local_parameterization);  // q = {x,y,z,w}
    }
    // only fix first state prior
    for (int i=0;i<1;i++)
    {
        cost_function = new ceres::AutoDiffCostFunction<prior_functor, 7, 3,4,3,4>(new prior_functor);
        problem.AddResidualBlock(cost_function,NULL,&p[i][0],&q[i][0],&p0[i][0],&q0[i][0]);

        problem.SetParameterBlockConstant(&p0[i][0]);
        problem.SetParameterBlockConstant(&q0[i][0]);
    }

    // add acc & omega constraint   ----  Attention! Probably needs a less weight
    // t \in [ head+1, head+2 )
    int imu_cnt = 0;
    vector<double*> imu_info_list;
    imu_info_list.clear();

    int l = graph.state[head+2].imu_data.size();  // [head+1, head+2) imu_data be stored in state[head+2]
    for (int i=0;i<l;i++)
    {
        imu_info_list.push_back( (double*)malloc(sizeof(double)*7) );  // malloc 1+3+3 double  (u,acc,omega)
        *(imu_info_list[imu_cnt]) = (graph.state[head+2].imu_data[i].ts - graph.state[head+1].stamp)/deltaT;
        *(imu_info_list[imu_cnt]+1) = graph.state[head+2].imu_data[i].a[0];
        *(imu_info_list[imu_cnt]+2) = graph.state[head+2].imu_data[i].a[1];
        *(imu_info_list[imu_cnt]+3) = graph.state[head+2].imu_data[i].a[2];
        *(imu_info_list[imu_cnt]+4) = graph.state[head+2].imu_data[i].w[0];
        *(imu_info_list[imu_cnt]+5) = graph.state[head+2].imu_data[i].w[1];
        *(imu_info_list[imu_cnt]+6) = graph.state[head+2].imu_data[i].w[2];

        cost_function = new ceres::AutoDiffCostFunction<acc_functor, 3, 3,4,3,4,3,4,3,4, 7>(new acc_functor);
        problem.AddResidualBlock(cost_function,NULL, &p[0][0],&q[0][0],&p[1][0],&q[1][0],&p[2][0],&q[2][0],&p[3][0],&q[3][0],
                                                     imu_info_list[imu_cnt]
                                );

        cost_function = new ceres::AutoDiffCostFunction<omega_functor, 3, 3,4,3,4,3,4,3,4, 7>(new omega_functor);
        problem.AddResidualBlock(cost_function,NULL, &p[0][0],&q[0][0],&p[1][0],&q[1][0],&p[2][0],&q[2][0],&p[3][0],&q[3][0],
                                                     imu_info_list[imu_cnt]
                                );

        problem.SetParameterBlockConstant(imu_info_list[imu_cnt]);

        imu_cnt++;
    }

    double key_frame_RT[7];  // [p,q.x(),q.y(),q.z(),q.w()]
    key_frame_RT[0] = graph.state[key_frame_no].p[0];
    key_frame_RT[1] = graph.state[key_frame_no].p[1];
    key_frame_RT[2] = graph.state[key_frame_no].p[2];
    key_frame_RT[3] = graph.state[key_frame_no].q.x();
    key_frame_RT[4] = graph.state[key_frame_no].q.y();
    key_frame_RT[5] = graph.state[key_frame_no].q.z();
    key_frame_RT[6] = graph.state[key_frame_no].q.w();

#ifdef UNBLUR
    // add blur-VO constraint
    // transform default col-major Eigen matrix to row-major, to form ceres::Grid2d
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> rowmajor_img = graph.state[head+2].img_data[calc_level];  // current frame
    ceres::Grid2D<double,1> array(rowmajor_img.data(),0,rowmajor_img.rows(),0,rowmajor_img.cols());
    ceres::BiCubicInterpolator< ceres::Grid2D<double,1> > interpolator(array);

    // dvo-functor
    for (int v=1;v<graph.state[key_frame_no].img_data[calc_level].rows()-1;v++)
        for (int u=1;u<graph.state[key_frame_no].img_data[calc_level].cols()-1;u++)
        {
            cost_function = new ceres::AutoDiffCostFunction<dvo_functor, 1, 3,4,3,4,3,4,3,4, 7>
                                (
                                 new dvo_functor(
                                                     graph.state[key_frame_no].img_data[calc_level],graph.state[key_frame_no].depth[calc_level],interpolator,
                                                     graph.state[key_frame_no].img_data[calc_level].rows(),graph.state[key_frame_no].img_data[calc_level].cols(),
                                                     cali.fx[calc_level], cali.fy[calc_level], cali.cx[calc_level], cali.cy[calc_level],
                                                     cali.exposure_time/deltaT,
                                                     u,v
                                                    )
                                );
            problem.AddResidualBlock(cost_function,NULL, &p[0][0],&q[0][0],&p[1][0],&q[1][0],&p[2][0],&q[2][0],&p[3][0],&q[3][0],
                                                         key_frame_RT
                                    );
        }
#else
    // add blur-VO constraint
    // transform default col-major Eigen matrix to row-major, to form ceres::Grid2d
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> rowmajor_img = graph.state[head+2].img_data[calc_level];
    ceres::Grid2D<double,1> array(rowmajor_img.data(),0,rowmajor_img.rows(),0,rowmajor_img.cols());
    ceres::BiCubicInterpolator< ceres::Grid2D<double,1> > interpolator(array);

    // blur-vo functor
    for (int v=1;v<graph.state[key_frame_no].img_data[calc_level].rows()-1;v++)
        for (int u=1;u<graph.state[key_frame_no].img_data[calc_level].cols()-1;u++)
        {
            cost_function = new ceres::AutoDiffCostFunction<blur_vo_functor, 1, 3,4,3,4,3,4,3,4, 7>
                                (
                                 new blur_vo_functor(
                                                     graph.state[key_frame_no].img_data[calc_level],graph.state[key_frame_no].depth[calc_level],interpolator,
                                                     graph.state[key_frame_no].img_data[calc_level].rows(),graph.state[key_frame_no].img_data[calc_level].cols(),
                                                     cali.fx[calc_level], cali.fy[calc_level], cali.cx[calc_level], cali.cy[calc_level],
                                                     cali.exposure_time/deltaT,
                                                     u,v
                                                    )
                                );
            problem.AddResidualBlock(cost_function,NULL, &p[0][0],&q[0][0],&p[1][0],&q[1][0],&p[2][0],&q[2][0],&p[3][0],&q[3][0],
                                                         key_frame_RT
                                    );
        }
#endif

    problem.SetParameterBlockConstant(key_frame_RT);


    // solving
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 7;
    options.line_search_sufficient_function_decrease = 1e-4;
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

        SE3_vec[i] = Sophus::SE3d(quat.toRotationMatrix(),T);
    }
}

// can only update pose [head, head+1, head+2, head+3]
// and velocity (head+2)    -> head+2 use the last ts<head+2 to approximate
void update_output_result(int head, Eigen::MatrixXd& est_img)
{
    // update to graph
    for (int i=0;i<4;i++)
    {
        graph.state[head+i].p = SE3_vec[i].translation();
        graph.state[head+i].q = Eigen::Quaterniond(SE3_vec[i].rotationMatrix());
    }

    // ---- construct B ----
    Eigen::Matrix<double,4,4> B;
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

    Eigen::Matrix<double,4,4> tmp_B;
    tmp_B = 1.0/6.0 * B;
    B = tmp_B;
    // --------------------

    Eigen::Vector3d last_linear_vel;  // in body frame, approximate to graph.state[head+2].v

    // output result to file
    double diff = 0.001;  // 1ms intepolate
    double exposure_begin_stamp = graph.state[head+2].stamp - cali.exposure_time;
    Eigen::MatrixXd AI_cnt;
    est_img = Eigen::MatrixXd::Zero(cali.height[calc_level],cali.width[calc_level]);
    AI_cnt = Eigen::MatrixXd::Zero(cali.height[calc_level],cali.width[calc_level]);

    Sophus::SE3d RTl0 = SE3_vec[0];
    for (double ts=graph.state[head+1].stamp;ts<graph.state[head+2].stamp;ts+=diff)
    {
        double u = (ts-graph.state[head+1].stamp)/deltaT;
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
            Eigen::VectorXd upsilon_omega = Sophus::SE3d::log(SE3_vec[j-1].inverse() * SE3_vec[j]);
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

        last_linear_vel = R.transpose() * linear_vel;  // transform world frame vel to body frame

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

        Eigen::Vector3d spline_acc = R.transpose() * ddSE.block<3,1>(0,3);  // ? g0 has been removed

        fprintf(solve_acc_file,"%lf %lf %lf %lf\n",
                                ts,spline_acc(0),spline_acc(1),spline_acc(2)
               );

        // --- generate est img ---
        if (ts>=exposure_begin_stamp)
        {
            Sophus::SE3d SE3_j(R,T);
            // SE(3)_i^j = ( SE(3)_j^0 )^-1 * SE(3)_i^0
            Sophus::SE3d key_frame_SE3(graph.state[key_frame_no].q.toRotationMatrix(),graph.state[key_frame_no].p);
            Sophus::SE3d SE3_i_2_j = SE3_j.inverse() * key_frame_SE3;

            int height,width;
            height = cali.height[calc_level];
            width = cali.width[calc_level];
            for (int v=1;v<height-1;v++)
                for (int u=1;u<width-1;u++)
                {
                    Eigen::Matrix<double,3,1> p_ref,p_cur;  // [x,y,z]
                    double lambda;
                    lambda = graph.state[key_frame_no].depth[calc_level](v,u);

                    if (double_equ_check(lambda,0.0,DOUBLE_EPS)<=0) // no depth
                        continue;

                    p_ref[0] = (u-cali.cx[calc_level])/cali.fx[calc_level] * lambda;
                    p_ref[1] = (v-cali.cy[calc_level])/cali.fy[calc_level] * lambda;
                    p_ref[2] = lambda;

                    p_cur = SE3_i_2_j * p_ref;

                    double u_new,v_new;
                    u_new = (p_cur[0]/p_cur[2]) * cali.fx[calc_level] + cali.cx[calc_level];
                    v_new = (p_cur[1]/p_cur[2]) * cali.fy[calc_level] + cali.cy[calc_level];

                    if (u_new < 1 || u_new >= width-1 || v_new < 1 || v_new >= height-1)
                        continue;

                    int iv,iu;
                    iv = v_new;
                    iu = u_new;

                    est_img(iv,iu) = est_img(iv,iu) + graph.state[key_frame_no].img_data[calc_level](v,u);
                    AI_cnt(iv,iu) = AI_cnt(iv,iu) + 1.0;
                }
        }
    }

    graph.state[head+2].v = last_linear_vel;

    int height,width;
    height = cali.height[calc_level];
    width = cali.width[calc_level];

#ifdef UNBLUR
    est_img = Eigen::MatrixXd::Zero(cali.height[calc_level],cali.width[calc_level]);
    AI_cnt = Eigen::MatrixXd::Zero(cali.height[calc_level],cali.width[calc_level]);

    // SE(3)_i^j = ( SE(3)_j^0 )^-1 * SE(3)_i^0
    Sophus::SE3d key_frame_SE3(graph.state[key_frame_no].q.toRotationMatrix(),graph.state[key_frame_no].p);
    Sophus::SE3d SE3_i_2_j = SE3_vec[2].inverse() * key_frame_SE3;
    for (int v=1;v<height-1;v++)
        for (int u=1;u<width-1;u++)
        {
            Eigen::Matrix<double,3,1> p_ref,p_cur;  // [x,y,z]
            double lambda;
            lambda = graph.state[key_frame_no].depth[calc_level](v,u);

            if (double_equ_check(lambda,0.0,DOUBLE_EPS)<=0) // no depth
                continue;

            p_ref[0] = (u-cali.cx[calc_level])/cali.fx[calc_level] * lambda;
            p_ref[1] = (v-cali.cy[calc_level])/cali.fy[calc_level] * lambda;
            p_ref[2] = lambda;

            p_cur = SE3_i_2_j * p_ref;

            double u_new,v_new;
            u_new = (p_cur[0]/p_cur[2]) * cali.fx[calc_level] + cali.cx[calc_level];
            v_new = (p_cur[1]/p_cur[2]) * cali.fy[calc_level] + cali.cy[calc_level];

            if (u_new < 1 || u_new >= width-1 || v_new < 1 || v_new >= height-1)
                continue;

            int iv,iu;
            iv = v_new;
            iu = u_new;

            est_img(iv,iu) = est_img(iv,iu) + graph.state[key_frame_no].img_data[calc_level](v,u);
            AI_cnt(iv,iu) = AI_cnt(iv,iu) + 1.0;
        }
#endif

    // generate img
    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
        {
            if (double_equ_check(AI_cnt(v,u),0.0,DOUBLE_EPS)<=0)
                est_img(v,u) = 0.0;
            else
                est_img(v,u) = est_img(v,u) / AI_cnt(v,u);
        }
}

void ceres_process(int head, Eigen::MatrixXd& est_img)
{
    SE3_vec.clear();
    for (int i=head;i<head+4;i++)
    {
        Sophus::SE3d tmp_SE3(graph.state[i].q.toRotationMatrix(),graph.state[i].p);
        SE3_vec.push_back(tmp_SE3);
    }

    ceres_solve(head);

    // ---- output to file ----
    update_output_result(head,est_img);
}
