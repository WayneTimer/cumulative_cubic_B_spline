# cumulative_cubic_B_spline

1. Cumulative cubic B-spline fitting done, correct.
2. Use <Spline-fusion>'s first derivates of SE(3) to get angular velocity, correct.
3. second derivates to get acc, not pretty correct.

TODO:
1. Add (RIC,TIC).
2. Try to use Ceres AD to get the first derivates and compare with /omega from IMU.
