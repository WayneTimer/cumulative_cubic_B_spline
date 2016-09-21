%% declare

%% plot  n x m  subplot
n = 4;
m = 3;

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

scatter_size = 10;

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

%% plot B-spline first order derivative
spline_vel_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_vel.txt';

data = load(spline_vel_file);

stamp = data(:,1);
omega = data(:,2:4);

subplot(n,m,7);
plot(stamp,omega(:,1),'r');
hold on;
title('\omega^k (0) - yaw');
subplot(n,m,8);
plot(stamp,omega(:,2),'r');
hold on;
title('\omega^k (1) - pitch');
subplot(n,m,9);
plot(stamp,omega(:,3),'r');
hold on;
title('\omega^k (2) - roll');

%% plot gt vel
debug_vel_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/debug_gt_vel.txt';
debug_vel_data = load(debug_vel_file);

stamp = debug_vel_data(:,1);
debug_omega = debug_vel_data(:,2:4);

subplot(n,m,7);
scatter(stamp,debug_omega(:,1),scatter_size);
subplot(n,m,8);
scatter(stamp,debug_omega(:,2),scatter_size);
subplot(n,m,9);
scatter(stamp,debug_omega(:,3),scatter_size);

%% plot B-spline second order derivative
spline_acc_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/spline_acc.txt';

data = load(spline_acc_file);

stamp = data(:,1);
acc = data(:,2:4);

subplot(n,m,10);
plot(stamp,acc(:,1),'r');
hold on;
title('acc^k (0)');
subplot(n,m,11);
plot(stamp,acc(:,2),'r');
hold on;
title('acc^k (1)');
subplot(n,m,12);
plot(stamp,acc(:,3),'r');
hold on;
title('acc^k (2)');

%% plot gt vel
debug_acc_file = '/home/timer/catkin_ws/src/cumulative_cubic_B_spline/helper/matlab_src/B_spline_plot/debug_gt_acc.txt';
debug_acc_data = load(debug_acc_file);

stamp = debug_acc_data(:,1);
debug_acc = debug_acc_data(:,2:4);

subplot(n,m,10);
scatter(stamp,debug_acc(:,1),scatter_size);
subplot(n,m,11);
scatter(stamp,debug_acc(:,2),scatter_size);
subplot(n,m,12);
scatter(stamp,debug_acc(:,3),scatter_size);