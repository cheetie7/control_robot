import time
import numpy as np

import rclpy
from sensor_msgs.msg import JointState
from factr_teleop.factr_teleop import FACTRTeleop

def create_array_msg(data):
    msg = JointState()
    msg.position = list(map(float, np.atleast_1d(data)))
    return msg

class Factr_teleop_Ros2(FACTRTeleop):
    def __init__(self):
        super().__init__()
        self.gripper_feedback_gain = self.config["controller"]["gripper_feedback"]["gain"]
        self.gripper_torque_ema_beta = self.config["controller"]["gripper_feedback"]["ema_beta"]
        self.gripper_external_torque = 0.0
        self.external_arm_torque = None
    def set_up_communication(self):
        self.arm_pos_publisher=self.create_publisher(JointState,f'{self.name}/arm_pos',10)
        self.gripper_pos_publisher=self.create_publisher(JointState,f'{self.name}/gripper_pos',10)
        if self.enable_torque_feedback:
            self.torque_feedback_sub=self.create_subscription(JointState,f'{self.name}/torque',self.arm_external_torque_callback,10)
        if self.enable_gripper_feedback:
            self.gripper_torque_feedback_sub=self.create_subscription(JointState,f'{self.name}/gripper_torque',self._gripper_external_torque_callback,10)
            
    def get_leader_arm_external_joint_torque(self):
        if self.external_arm_torque is None:
            return np.zeros(self.num_arm_joints)
        return self.external_arm_torque

    def arm_external_torque_callback(self,msg):
        self.external_arm_torque = np.array(msg.position, dtype=float)

    def _gripper_external_torque_callback(self, msg):
        
        gripper_external_torque = msg.position[0]
        self.gripper_external_torque = self.gripper_torque_ema_beta * self.gripper_external_torque + \
            (1-self.gripper_torque_ema_beta) * gripper_external_torque
    def gripper_feedback(self, leader_gripper_pos, leader_gripper_vel, gripper_feedback):
        torque_gripper = -1.0*gripper_feedback / self.gripper_feedback_gain
        return torque_gripper
    def get_leader_gripper_feedback(self):
        return self.gripper_external_torque

    def update_communication(self, leader_arm_pos, leader_gripper_pos):
        self.arm_pos_publisher.publish(create_array_msg(leader_arm_pos))
        self.gripper_pos_publisher.publish(create_array_msg([leader_gripper_pos]))
        
def main(args=None):
    rclpy.init(args=args)
    rosnode=Factr_teleop_Ros2()
    try:
        while rclpy.ok():
            rclpy.spin(rosnode)
    except KeyboardInterrupt:
        rosnode.get_logger().info('键盘控制停止')
        rosnode.shut_down()
    finally:
        rclpy.shutdown()
