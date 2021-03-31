#!/usr/bin/env python3
import numpy as np 
import math
import nvector as nv
import cv2 

deg = math.pi/180. 
G = 9.81007
EPSILON = 1e-6

def skew_matrix(v):
    return np.array([[0., -v[2], v[1]],[v[2], 0., -v[0]],[-v[1], v[0], 0.]])

class State:
    def __init__(self):
        self.timestamp = 0.
        self.cov = np.zeros((15,15))
        self.p_GI = np.array([.0,.0,.0])
        self.v_GI = np.array([.0,.0,.0])
        self.r_GI = np.eye(3)
        self.acc_bias = np.array([.0,.0,.0])
        self.gyr_bias = np.array([.0,.0,.0])


class EKF:
    def __init__(self,acc_n,gyr_n,acc_w,gyr_w):
        self.acc_noise_ = acc_n
        self.gyr_noise_ = gyr_n
        self.acc_bias_noise_ = acc_w
        self.gyr_bias_noise_ = gyr_w
        self.state = State()

        sigma_rp = 10*deg 
        sigma_yaw = 100*deg

        self.state.cov[:3,:3] = np.eye(3)*100
        self.state.cov[3:6,3:6] = np.eye(3)*100
        self.state.cov[6:8,6:8] = np.eye(2)*sigma_rp*sigma_rp
        self.state.cov[8,8] = sigma_yaw * sigma_yaw
        self.state.cov[9:12,9:12] = np.eye(3) * 0.0004
        self.state.cov[12:15,12:15] = np.eye(3)*0.0004

    def predict(self, last_imu, curr_imu):
        dt = curr_imu.timestamp - last_imu.timestamp
        dt2 = dt*dt 
        
        last_state = self.state
        self.state.timestamp = curr_imu.timestamp

        acc_unbias = 0.5 * (last_imu.acc+curr_imu.acc) - last_state.acc_bias
        gyr_unbias = 0.5 * (last_imu.gyr + curr_imu.gyr) - last_state.gyr_bias
        acc_nominal = last_state.r_GI.dot(acc_unbias) + np.array([0,0,-G])
        self.state.p_GI = last_state.p_GI + last_state.v_GI*dt + 0.5*acc_nominal*dt2
        self.state.v_GI = last_state.v_GI + acc_nominal * dt
        delta_angle_axis = gyr_unbias*dt 
        norm_delta_angle = np.linalg.norm(delta_angle_axis)
        dR = np.eye(3)
        if(norm_delta_angle>EPSILON):
            dR = cv2.Rodrigues(delta_angle_axis)[0]
            self.state.r_GI = last_state.r_GI.dot(dR)
        Fx = np.eye(15)
        Fx[:3,3:6] = np.eye(3)*dt
        Fx[3:6,6:9] = -self.state.r_GI.dot(skew_matrix(acc_unbias))*dt 
        Fx[3:6,9:12] = -self.state.r_GI * dt 
        if(norm_delta_angle>EPSILON):
            Fx[6:9,6:9] = dR.T 
        else:
            Fx[6:9,6:9] = np.eye(3)
        Fx[6:9,12:15] = -np.eye(3)*dt 

        Fi = np.zeros((15,12))
        Fi[3:15,0:12] = np.eye(12)

        Qi = np.zeros((12,12))
        Qi[:3,:3] = np.eye(3)*dt2*self.acc_noise_
        Qi[3:6,3:6] = np.eye(3)*dt2*self.gyr_noise_
        Qi[6:9,6:9] = np.eye(3)*dt*self.acc_bias_noise_
        Qi[9:12,9:12] = np.eye(3)*dt*self.gyr_bias_noise_

        self.state.cov = Fx.dot(last_state.cov).dot(Fx.T) + Fi.dot(Qi).dot(Fi.T)


    def update_measurement(self,H,V,r):
       # import ipdb;ipdb.set_trace()
        P = self.state.cov 
        S = H.dot(P).dot(H.T) + V 
        K = P.dot(H.T).dot(np.linalg.inv(S))

        delta_x = K.dot(r)
        self.state.p_GI += delta_x[:3]
        self.state.v_GI += delta_x[3:6]
        dR = delta_x[6:9]
        if(np.linalg.norm(dR)>EPSILON):
            self.state.r_GI = self.state.r_GI.dot(cv2.Rodrigues(dR)[0])
        self.state.acc_bias += delta_x[9:12]
        self.state.gyr_bias += delta_x[12:15]
        I_KH = np.eye(15) - K.dot(H)
        self.state.cov = I_KH.dot(P).dot(I_KH.T) + K.dot(V).dot(K.T)

    
    




    