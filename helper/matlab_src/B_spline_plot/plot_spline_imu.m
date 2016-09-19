%% declare

%% plot  n x m  subplot
n = 2;
m = 3;

%% plot multi-spine
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