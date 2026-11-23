"""
Decentralized CoDe-CBF ROS 2 Humble Safety Filter Node.
Runs at 50 Hz, tracks DDS Age-of-Information (AoI), solves OSQP QP,
and commands safe acceleration setpoints for PX4 Offboard control.
"""

import sys
sys.path.append("/home/jarvis/Drones_CoDe_CBF")

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Accel, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import time, json

from src.code_cbf.cbf_qp_filter import CoDeCBFFilter

class CoDeCBFNode(Node):
    def __init__(self):
        super().__init__('code_cbf_safety_filter')
        
        # ROS 2 Parameters
        self.declare_parameter('uav_id', 1)
        self.declare_parameter('total_uavs', 4)
        self.declare_parameter('r_safe', 0.8)
        self.declare_parameter('kappa_z', 2.0)
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('alpha_target', 0.05)
        self.declare_parameter('lambda_aoi', 1.5)
        
        self.uav_id = self.get_parameter('uav_id').value
        self.total_uavs = self.get_parameter('total_uavs').value
        self.r_safe = self.get_parameter('r_safe').value
        self.kappa_z = self.get_parameter('kappa_z').value
        self.dt = self.get_parameter('dt').value
        
        # Instantiate CoDe-CBF core engine
        self.filter = CoDeCBFFilter(
            r_safe=self.r_safe,
            kappa_z=self.kappa_z,
            dt=self.dt,
            alpha_target=self.get_parameter('alpha_target').value,
            lambda_aoi=self.get_parameter('lambda_aoi').value
        )
        
        # Ego state
        self.ego_pos = np.zeros(3)
        self.ego_vel = np.zeros(3)
        self.u_nom = np.zeros(3)
        self.goal_pos = np.array([5.0, 0.0, 1.5])
        
        # Neighbor states & AoI trackers: map other_id -> {'pos': ..., 'vel': ..., 'aoi_ticks': ...}
        self.neighbors = {}
        
        # Subscribers
        self.sub_ego_odom = self.create_subscription(
            Odometry, f'/uav_{self.uav_id}/odometry', self.ego_odom_callback, 10
        )
        self.sub_cmd_nom = self.create_subscription(
            Accel, f'/uav_{self.uav_id}/cmd_acc_nom', self.cmd_nom_callback, 10
        )
        
        # Subscriptions for neighbor odometries over ROS 2 DDS
        self.neighbor_subs = []
        for other_id in range(1, self.total_uavs + 1):
            if other_id != self.uav_id:
                sub = self.create_subscription(
                    Odometry,
                    f'/uav_{other_id}/odometry',
                    lambda msg, o_id=other_id: self.neighbor_odom_callback(msg, o_id),
                    10
                )
                self.neighbor_subs.append(sub)
                
        # Publishers
        self.pub_safe_acc = self.create_publisher(
            Accel, f'/uav_{self.uav_id}/cmd_acc_safe', 10
        )
        self.pub_telemetry = self.create_publisher(
            String, f'/uav_{self.uav_id}/cbf_telemetry', 10
        )
        
        # 50 Hz control loop timer (0.02s)
        self.timer = self.create_timer(self.dt, self.control_loop_callback)
        self.get_logger().info(f'CoDe-CBF Safety Filter Node for UAV {self.uav_id} initialized at 50 Hz.')

    def ego_odom_callback(self, msg: Odometry):
        self.ego_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.ego_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])

    def cmd_nom_callback(self, msg: Accel):
        self.u_nom = np.array([msg.linear.x, msg.linear.y, msg.linear.z])

    def neighbor_odom_callback(self, msg: Odometry, other_id: int):
        # Calculate real packet delay / Age-of-Information
        recv_now = self.get_clock().now().nanoseconds * 1e-9
        stamp_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        delay_sec = max(0.0, recv_now - stamp_time)
        aoi_ticks = int(np.round(delay_sec / self.dt))
        
        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        self.neighbors[other_id] = {
            'id': other_id,
            'pos': pos,
            'vel': vel,
            'aoi_ticks': aoi_ticks
        }

    def control_loop_callback(self):
        neighbors_list = list(self.neighbors.values())
        u_safe, delta_slack, solve_ms, telem = self.filter.filter_control(
            p_i=self.ego_pos,
            v_i=self.ego_vel,
            u_nom=self.u_nom,
            p_goal_i=self.goal_pos,
            neighbors=neighbors_list
        )
        
        # Publish filtered safe acceleration
        out_msg = Accel()
        out_msg.linear.x = float(u_safe[0])
        out_msg.linear.y = float(u_safe[1])
        out_msg.linear.z = float(u_safe[2])
        self.pub_safe_acc.publish(out_msg)
        
        # Publish telemetry JSON string
        telem_msg = String()
        telem_data = {
            'uav_id': self.uav_id,
            'solve_ms': solve_ms,
            'delta_slack': delta_slack,
            'status': telem['status'],
            'active_neighbors': len(neighbors_list)
        }
        telem_msg.data = json.dumps(telem_data)
        self.pub_telemetry.publish(telem_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CoDeCBFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
