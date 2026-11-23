"""
3D ROS 2 RViz Swarm Visualizer Node for CoDe-CBF.
Publishes:
1. 3D Quadrotor Meshes & Body Frames (/swarm/markers/drones)
2. Downwash Anisotropic Ellipsoidal Barriers E = diag(1, 1, 1/kappa_z^2) (/swarm/markers/downwash_ellipsoids)
3. Dynamic Trajectory Ribbons (/swarm/markers/trajectories)
4. Active CBF Acceleration & Deadlock Rotation Vectors (/swarm/markers/control_vectors)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np

class Swarm3DVisualizer(Node):
    def __init__(self):
        super().__init__('swarm_3d_visualizer')
        
        self.declare_parameter('total_uavs', 4)
        self.declare_parameter('r_safe', 0.8)
        self.declare_parameter('kappa_z', 2.0)
        
        self.total_uavs = self.get_parameter('total_uavs').value
        self.r_safe = self.get_parameter('r_safe').value
        self.kappa_z = self.get_parameter('kappa_z').value
        
        # State tracking
        self.uav_positions = {}
        self.uav_trajectories = {i: [] for i in range(1, self.total_uavs + 1)}
        
        # Subscriptions
        self.subs = []
        for i in range(1, self.total_uavs + 1):
            sub = self.create_subscription(
                Odometry,
                f'/uav_{i}/odometry',
                lambda msg, u_id=i: self.odom_callback(msg, u_id),
                10
            )
            self.subs.append(sub)
            
        # Publishers
        self.pub_drone_markers = self.create_publisher(MarkerArray, '/swarm/markers/drones', 10)
        self.pub_barrier_markers = self.create_publisher(MarkerArray, '/swarm/markers/downwash_ellipsoids', 10)
        self.pub_traj_markers = self.create_publisher(MarkerArray, '/swarm/markers/trajectories', 10)
        
        # 20 Hz visualization timer
        self.timer = self.create_timer(0.05, self.publish_visualization)
        self.colors = [
            (0.12, 0.47, 0.71), # Blue
            (0.84, 0.15, 0.16), # Red
            (0.17, 0.63, 0.17), # Green
            (1.00, 0.50, 0.05), # Orange
            (0.58, 0.40, 0.74), # Purple
            (0.55, 0.34, 0.29)  # Brown
        ]

    def odom_callback(self, msg: Odometry, u_id: int):
        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.uav_positions[u_id] = pos
        self.uav_trajectories[u_id].append(pos.copy())
        if len(self.uav_trajectories[u_id]) > 500:
            self.uav_trajectories[u_id].pop(0)

    def publish_visualization(self):
        now = self.get_clock().now().to_msg()
        drone_array = MarkerArray()
        barrier_array = MarkerArray()
        traj_array = MarkerArray()
        
        for u_id, pos in self.uav_positions.items():
            c_rgb = self.colors[(u_id - 1) % len(self.colors)]
            
            # 1. Drone Body Marker (Sphere)
            m_drone = Marker()
            m_drone.header.frame_id = "world"
            m_drone.header.stamp = now
            m_drone.ns = "drones"
            m_drone.id = u_id
            m_drone.type = Marker.SPHERE
            m_drone.action = Marker.ADD
            m_drone.pose.position.x = float(pos[0])
            m_drone.pose.position.y = float(pos[1])
            m_drone.pose.position.z = float(pos[2])
            m_drone.pose.orientation.w = 1.0
            m_drone.scale.x = 0.35 # 35cm quadrotor diameter
            m_drone.scale.y = 0.35
            m_drone.scale.z = 0.12
            m_drone.color = ColorRGBA(r=c_rgb[0], g=c_rgb[1], b=c_rgb[2], a=0.95)
            drone_array.markers.append(m_drone)
            
            # 2. Downwash Anisotropic Ellipsoidal Barrier Marker
            # Semi-axes: rx = R_safe, ry = R_safe, rz = kappa_z * R_safe
            m_barrier = Marker()
            m_barrier.header.frame_id = "world"
            m_barrier.header.stamp = now
            m_barrier.ns = "barriers"
            m_barrier.id = u_id + 100
            m_barrier.type = Marker.SPHERE
            m_barrier.action = Marker.ADD
            m_barrier.pose.position.x = float(pos[0])
            m_barrier.pose.position.y = float(pos[1])
            m_barrier.pose.position.z = float(pos[2])
            m_barrier.pose.orientation.w = 1.0
            m_barrier.scale.x = 2.0 * self.r_safe # diameter
            m_barrier.scale.y = 2.0 * self.r_safe
            m_barrier.scale.z = 2.0 * self.r_safe * self.kappa_z # vertical elongation
            m_barrier.color = ColorRGBA(r=c_rgb[0], g=c_rgb[1], b=c_rgb[2], a=0.22)
            barrier_array.markers.append(m_barrier)
            
            # 3. Trajectory Ribbon
            m_traj = Marker()
            m_traj.header.frame_id = "world"
            m_traj.header.stamp = now
            m_traj.ns = "trajectories"
            m_traj.id = u_id + 200
            m_traj.type = Marker.LINE_STRIP
            m_traj.action = Marker.ADD
            m_traj.scale.x = 0.04 # 4cm line thickness
            m_traj.color = ColorRGBA(r=c_rgb[0], g=c_rgb[1], b=c_rgb[2], a=0.7)
            m_traj.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in self.uav_trajectories[u_id]]
            traj_array.markers.append(m_traj)
            
        if drone_array.markers:
            self.pub_drone_markers.publish(drone_array)
            self.pub_barrier_markers.publish(barrier_array)
            self.pub_traj_markers.publish(traj_array)

def main(args=None):
    rclpy.init(args=args)
    node = Swarm3DVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
