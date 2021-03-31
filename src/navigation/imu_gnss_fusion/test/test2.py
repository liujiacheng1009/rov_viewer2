#!/usr/bin/env python3
import os,sys 
sys.path.append("/home/jc/Downloads/test_ws/src/imu_gnss_fusion/src")

from ekf import EKF
import numpy as np 


class ImuData:
    def __init__(self):
        self.timestamp = 0.0
        self.acc = np.array([.0,.0,.0])
        self.gyr = np.array([.0,.0,.0])

ekf = EKF(acc_n=1e-2,gyr_n=1e-4,acc_w=1e-6,gyr_w=1e-8)
last_imu = ImuData()
last_imu.timestamp = 1
curr_imu = ImuData()
curr_imu.timestamp = 1.1
curr_imu.acc = np.array([0.1,0,0.])
curr_imu.gyr = np.array([0.0,0.,0])
ekf.predict(last_imu,curr_imu)
H = np.array([[ 1.,0.,0.,0.,0.,0. ,-0. ,-0. ,-0.,0.,0.,0.,0.,0.,0.],
 [ 0.,1.,0.,0.,0.,0. ,-0. ,-0. ,-0.,0.,0.,0.,0.,0.,0.],
 [ 0.,0.,1.,0.,0.,0. ,-0., -0. ,-0.,0.,0.,0.,0.,0.,0.]])
V = np.array( [[0.002304  , 0.,       0.,      ],
        [0.,       0.00553536 ,0.,      ],
        [0.,       0.,       0.09884736]])
r = np.array( [1.88561492,0.08148084, 0.02500535])
ekf.update_measurement(H,V,r)
print("P_GI",ekf.state.p_GI)
print("v_GI",ekf.state.v_GI)
print("r_GI",ekf.state.r_GI)
print("cov",ekf.state.cov)
