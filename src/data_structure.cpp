#include "data_structure.h"
#include "utils.h"

State::State()
{
    p.setZero();
    q.setIdentity();
    v.setZero();
    alpha.setZero();
    beta.setZero();
    imu_data.clear();
    stamp = -1.0;
}

Graph::Graph()
{
    state.clear();
}

bool CALI_PARA::init(ros::NodeHandle& nh)
{
    // read from kalibr result
    XmlRpc::XmlRpcValue res;
    if (!nh.getParam("cam1",res))
    {
        puts("Can not get XmlRpcValue");
        return false;
    }
    width[0] = res["resolution"][0];
    height[0] = res["resolution"][1];
    fx[0] = res["intrinsics"][0];
    fy[0] = res["intrinsics"][1];
    cx[0] = res["intrinsics"][2];
    cy[0] = res["intrinsics"][3];
    baseline = res["T_cn_cnm1"][0][3];
    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
            R_I_2_C(i,j) = res["T_cam_imu"][i][j];
    for (int i=0;i<3;i++)
        T_I_2_C(i) = res["T_cam_imu"][i][3];

    // ---------
    // pyr_down: (fx,fy,cx,cy)
    for (int level=1;level<PYRDOWN_LEVEL;level++)
    {
        fx[level] = fx[0] / (1<<level);
        fy[level] = fy[0] / (1<<level);
        cx[level] = (cx[0]+0.5) / (1<<level) - 0.5;
        cy[level] = (cy[0]+0.5) / (1<<level) - 0.5;
        width[level] = width[0] >> level;
        height[level] = height[0] >> level;
    }
    return true;
}

void CALI_PARA::view()
{
    // output params to view
    printf("cali.width[0] = %d\n",width[0]);
    printf("cali.height[0] = %d\n",height[0]);
    printf("cali.fx[0] = %lf\n",fx[0]);
    printf("cali.fy[0] = %lf\n",fy[0]);
    printf("cali.cx[0] = %lf\n",cx[0]);
    printf("cali.cy[0] = %lf\n",cy[0]);
    printf("cali.baseline = %lf\n",baseline);
    cout << "cali.R_I_2_C:" << endl;
    cout << R_I_2_C << endl;
    cout << "cali.T_I_2_C:" << endl;
    cout << T_I_2_C.transpose() << endl;
}
