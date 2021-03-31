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
ekf.update_measurement()
# print(ekf.state.p_GI)
# print(ekf.state.v_GI)
# print(ekf.state.r_GI)
#print(ekf.state.cov)