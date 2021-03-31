#!/usr/bin/env python3
import rospy
from ekf import EKF,skew_matrix
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import PoseStamped,Point, Pose, Quaternion, Twist, Vector3
import numpy as np 
import nvector as nv
from collections import deque
from nvector import rad, deg

fixed_id = "world"

class ImuData:
    def __init__(self):
        self.timestamp = 0.0
        self.acc = np.array([.0,.0,.0])
        self.gyr = np.array([.0,.0,.0])

class GPSData:
    def __init__(self):
        self.timestamp = 0.0
        self.lla = np.array([.0,.0,.0])
        self.cov = np.eye(3)
    
    def lla2enu(self,init_lla,point_lla):
        n_EA_E = nv.lat_lon2n_E(rad(init_lla[0]), rad(init_lla[1]))
        n_EB_E = nv.lat_lon2n_E(rad(point_lla[0]), rad(point_lla[1]))
        p_AB_E = nv.n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, init_lla[2], point_lla[2])
        R_EN = nv.n_E2R_EN(n_EA_E)
        p_AB_N = np.dot(R_EN.T, p_AB_E).ravel()
        p_AB_N[0],p_AB_N[1] = p_AB_N[1],p_AB_N[0]
        return p_AB_N

    def enu2lla(self, init_lla, point_enu, point_lla):
        pass


class Fusion:
    def __init__(self, acc_n=1e-2,gyr_n=1e-4,acc_w=1e-6,gyr_w=1e-8):
        self.initialized = False
        self.init_lla_ = None
        self.I_p_Gps_ = np.array([.0,.0,.0])
        self.imu_buff = deque([],maxlen = 100)
        self.ekf = EKF(acc_n,gyr_n,acc_w,gyr_w)
        rospy.Subscriber("imu/data", Imu,self.imu_callback)
        rospy.Subscriber("/fix",NavSatFix, self.gnss_callback)
        self.pub_path = rospy.Publisher("nav_path",Path,queue_size=10)
        self.pub_odom = rospy.Publisher("nav_odom",Odometry, queue_size=10)
        self.nav_path = Path()
        self.last_imu = None
        self.p_G_Gps_= None

    def imu_callback(self,msg):
        imu = ImuData()
        imu.timestamp = msg.header.stamp.to_sec()
        imu.acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y,
                    msg.linear_acceleration.z])
        imu.gyr = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                    msg.angular_velocity.z])
        
        if(not self.initialized):
            self.imu_buff.append(imu)
            return
        self.ekf.predict(self.last_imu,imu)
        self.last_imu = imu 
        self.pub_state()

    def gnss_callback(self,msg):
        if(msg.status.status != 2):
            print("Error with GPS!!")
            return
        gps = GPSData()
        gps.timestamp = msg.header.stamp.to_sec()
        gps.lla = np.array([msg.latitude, msg.longitude, msg.altitude])
        gps.cov = np.array(msg.position_covariance).reshape((3,3))
        if(not self.initialized):
            if(len(self.imu_buff)<100):
                print("not enough imu data!!")
                return
            
            self.last_imu = self.imu_buff[99]
            if(abs(gps.timestamp-self.last_imu.timestamp)>0.5):
                print("GPS and imu not synchronized!!")
                return
            self.ekf.state.timestamp = self.last_imu.timestamp
            if(not self.init_rot_from_imu_data()):
                return
            self.init_lla_ = gps.lla
            self.initialized = True
            return

        self.p_G_Gps_ = gps.lla2enu(self.init_lla_, gps.lla)

        p_GI = self.ekf.state.p_GI
        r_GI = self.ekf.state.r_GI

        residual = self.p_G_Gps_-(p_GI+r_GI.dot(self.I_p_Gps_))
        H = np.zeros((3,15))
        H[:3,:3] = np.eye(3)
        H[:3,6:9] = -r_GI.dot(skew_matrix(self.I_p_Gps_))
        V = gps.cov
        self.ekf.update_measurement(H,V,residual)

    def init_rot_from_imu_data(self):
        ## 初始化姿态，加速的方差不能太大
        sum_acc = np.array([0.,0.,0.])
        for imu_data in self.imu_buff:
            sum_acc += imu_data.acc
        mean_acc = sum_acc /len(self.imu_buff)
        sum_err2 = np.array([0.,0.,0.])
        for imu_data in self.imu_buff:
            sum_err2 += np.power(imu_data.acc-mean_acc,2)
        std_acc = np.power(sum_err2/len(self.imu_buff),0.5)

        if(np.max(std_acc)>3.0):
            print("acc std is too big !!")
            return False
        
        ## 这里获得旋转矩阵的原理是？
        z_axis = mean_acc/np.linalg.norm(mean_acc)
        z_axis = z_axis.reshape((3,1))
        x_axis = np.array([1,0,0]).reshape((3,1))- z_axis.dot(z_axis.T).dot(np.array([1,0,0]).reshape((3,1)))
        x_axis = x_axis/np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis.reshape(3),x_axis.reshape(3)).reshape(3,1)
        y_axis = y_axis/np.linalg.norm(y_axis)

        r_IG = np.zeros((3,3))
        r_IG[:3,0] = x_axis.reshape(3)
        r_IG[:3,1] = y_axis.reshape(3)
        r_IG[:3,2] = z_axis.reshape(3)
        self.ekf.state.r_GI = r_IG.T ## 初始化姿态
        return True

    def pub_state(self):
        if(self.p_G_Gps_ is None):
            return
        odom_msg = Odometry()
        odom_msg.header.frame_id = fixed_id
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.pose.pose = Pose(Point(self.ekf.state.p_GI[0],self.ekf.state.p_GI[1],
                                    self.ekf.state.p_GI[2]),Quaternion(0,0,0,1))
        odom_msg.twist.twist = Twist(Vector3(self.ekf.state.v_GI[0],self.ekf.state.v_GI[1],
                                    self.ekf.state.v_GI[2]), Vector3(0,0,0))
        self.pub_odom.publish(odom_msg)

        pose_msg = PoseStamped()
        pose_msg.header = odom_msg.header
        pose_msg.pose = odom_msg.pose.pose
        self.nav_path.header = pose_msg.header
        self.nav_path.poses.append(pose_msg)
        self.pub_path.publish(self.nav_path)

if __name__ == "__main__":
    rospy.init_node("imu_gnss_fusion",anonymous=True)
    f = Fusion()
    rospy.spin()