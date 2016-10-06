%% declare

%% plot  n x m  subplot
% p:          x y z
% \theta:     yaw pitch roll
% vel:        x y z
% \omega:     yaw pitch roll
% acc:        x y z
% acc_bias:   x y z
% omega_bias: x y z
n = 7;
m = 3;
scatter_size = 5;

%% plot B-spline pose
spline_pose_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_pose.txt';

data = load(spline_pose_file);

stamp = data(:,1);
p = data(:,2:4);
theta = data(:,5:7);

subplot(n,m,1);
plot(stamp,p(:,1),'r');
hold on;
title('x- p_k^0');
subplot(n,m,2);
plot(stamp,p(:,2),'r');
hold on;
title('y - p_k^0');
subplot(n,m,3);
plot(stamp,p(:,3),'r');
hold on;
title('z - p_k^0');

subplot(n,m,4);
plot(stamp,theta(:,1),'r');
hold on;
title('\theta_k^0 (0) - yaw');
subplot(n,m,5);
plot(stamp,theta(:,2),'r');
hold on;
title('\theta_k^0 (1) - pitch');
subplot(n,m,6);
plot(stamp,theta(:,3),'r');
hold on;
title('\theta_k^0 (2) - roll');

%% debug plot gt pose
debug_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/debug_gt_pose.txt';
debug_data = load(debug_file);

stamp = debug_data(:,1);
p = debug_data(:,2:4);
theta = debug_data(:,5:7);
vel = debug_data(:,8:10);

subplot(n,m,1);
scatter(stamp,p(:,1),scatter_size);
subplot(n,m,2);
scatter(stamp,p(:,2),scatter_size);
subplot(n,m,3);
scatter(stamp,p(:,3),scatter_size);

subplot(n,m,4);
scatter(stamp,theta(:,1),scatter_size);
subplot(n,m,5);
scatter(stamp,theta(:,2),scatter_size);
subplot(n,m,6);
scatter(stamp,theta(:,3),scatter_size);

subplot(n,m,7);
scatter(stamp,vel(:,1),scatter_size);
hold on;
title('x- v_k^0');
subplot(n,m,8);
scatter(stamp,vel(:,2),scatter_size);
hold on;
title('y- v_k^0');
subplot(n,m,9);
scatter(stamp,vel(:,3),scatter_size);
hold on;
title('z- v_k^0');

%% plot B-spline first order derivative
spline_vel_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_vel.txt';

data = load(spline_vel_file);

stamp = data(:,1);
vel = data(:,2:4);

subplot(n,m,7);
plot(stamp,vel(:,1),'r');
subplot(n,m,8);
plot(stamp,vel(:,2),'r');
subplot(n,m,9);
plot(stamp,vel(:,3),'r');


%% plot B-spline \omega
spline_omega_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_omega.txt';

data = load(spline_omega_file);

stamp = data(:,1);
omega = data(:,2:4);

subplot(n,m,10);
plot(stamp,omega(:,1),'r');
hold on;
title('\omega^k (0) - yaw');
subplot(n,m,11);
plot(stamp,omega(:,2),'r');
hold on;
title('\omega^k (1) - pitch');
subplot(n,m,12);
plot(stamp,omega(:,3),'r');
hold on;
title('\omega^k (2) - roll');

%% plot imu
debug_imu_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/debug_imu.txt';
debug_imu_data = load(debug_imu_file);

stamp = debug_imu_data(:,1);
debug_acc = debug_imu_data(:,2:4);
debug_omega = debug_imu_data(:,5:7);

subplot(n,m,10);
plot(stamp,debug_omega(:,1));
subplot(n,m,11);
plot(stamp,debug_omega(:,2));
subplot(n,m,12);
plot(stamp,debug_omega(:,3));

subplot(n,m,13);
plot(stamp,debug_acc(:,1));
hold on;
title('x- acc^k (0)');
subplot(n,m,14);
plot(stamp,debug_acc(:,2));
hold on;
title('y- acc^k (1)');
subplot(n,m,15);
plot(stamp,debug_acc(:,3));
hold on;
title('z- acc^k (2)');

%% plot B-spline second order derivative
spline_acc_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_acc.txt';

data = load(spline_acc_file);

stamp = data(:,1);
acc = data(:,2:4);

subplot(n,m,13);
plot(stamp,acc(:,1),'r');
subplot(n,m,14);
plot(stamp,acc(:,2),'r');
subplot(n,m,15);
plot(stamp,acc(:,3),'r');

%% === plot solved B-spline's pose ===
solve_pose_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/solve_pose.txt';

data = load(solve_pose_file);

stamp = data(:,1);
p = data(:,2:4);
theta = data(:,5:7);

subplot(n,m,1);
plot(stamp,p(:,1),'g');
subplot(n,m,2);
plot(stamp,p(:,2),'g');
subplot(n,m,3);
plot(stamp,p(:,3),'g');

subplot(n,m,4);
plot(stamp,theta(:,1),'g');
subplot(n,m,5);
plot(stamp,theta(:,2),'g');
subplot(n,m,6);
plot(stamp,theta(:,3),'g');

%% === plot B-spline first order derivative ===
solve_vel_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/solve_vel.txt';

data = load(solve_vel_file);

stamp = data(:,1);
vel = data(:,2:4);

subplot(n,m,7);
plot(stamp,vel(:,1),'g');
subplot(n,m,8);
plot(stamp,vel(:,2),'g');
subplot(n,m,9);
plot(stamp,vel(:,3),'g');

%% === plot B-spline \omega ===
solve_omega_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/solve_omega.txt';

data = load(solve_omega_file);

stamp = data(:,1);
omega = data(:,2:4);

subplot(n,m,10);
plot(stamp,omega(:,1),'g');
subplot(n,m,11);
plot(stamp,omega(:,2),'g');
subplot(n,m,12);
plot(stamp,omega(:,3),'g');

%% === plot solved B-spline's acc ===
solve_acc_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/solve_acc.txt';

data = load(solve_acc_file);

stamp = data(:,1);
acc = data(:,2:4);

subplot(n,m,13);
plot(stamp,acc(:,1),'g');
subplot(n,m,14);
plot(stamp,acc(:,2),'g');
subplot(n,m,15);
plot(stamp,acc(:,3),'g');

%% === plot bias ===
bias_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/bias.txt';

data = load(bias_file);

stamp = data(:,1);
acc_bias = data(:,2:4);
omega_bias = data(:,5:7);

subplot(n,m,16);
plot(stamp,acc_bias(:,1),'g');
hold on;
title('x- acc bias (0)');
subplot(n,m,17);
plot(stamp,acc_bias(:,2),'g');
hold on;
title('y- acc bias (1)');
subplot(n,m,18);
plot(stamp,acc_bias(:,3),'g');
hold on;
title('z- acc bias (2)');

subplot(n,m,19);
plot(stamp,omega_bias(:,1),'g');
hold on;
title('x- omega bias (0)');
subplot(n,m,20);
plot(stamp,omega_bias(:,2),'g');
hold on;
title('y- omega bias (1)');
subplot(n,m,21);
plot(stamp,omega_bias(:,3),'g');
hold on;
title('z- omega bias (2)');