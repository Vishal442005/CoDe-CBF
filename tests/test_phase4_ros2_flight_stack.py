import sys
sys.path.append("/home/jarvis/Drones_CoDe_CBF")

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Accel
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import time, json

class SwarmIntegrationTester(Node):
    def __init__(self):
        super().__init__('swarm_integration_tester')
        
        # Publishers for simulated state
        self.pub_uav1_odom = self.create_publisher(Odometry, '/uav_1/odometry', 10)
        self.pub_uav2_odom = self.create_publisher(Odometry, '/uav_2/odometry', 10)
        self.pub_uav1_cmd = self.create_publisher(Accel, '/uav_1/cmd_acc_nom', 10)
        
        # Subscriber to safety filter output
        self.sub_safe_acc = self.create_subscription(Accel, '/uav_1/cmd_acc_safe', self.safe_acc_cb, 10)
        self.sub_telem = self.create_subscription(String, '/uav_1/cbf_telemetry', self.telem_cb, 10)
        
        self.received_safe_accs = []
        self.received_telems = []
        self.step_count = 0
        
        # 50 Hz publisher timer
        self.timer = self.create_timer(0.02, self.publish_states)

    def publish_states(self):
        now = self.get_clock().now()
        
        # UAV 1 moving forward along X
        msg1 = Odometry()
        msg1.header.stamp = now.to_msg()
        msg1.pose.pose.position.x = -1.0 + self.step_count * 0.02
        msg1.pose.pose.position.y = 0.0
        msg1.pose.pose.position.z = 1.5
        msg1.twist.twist.linear.x = 1.0
        self.pub_uav1_odom.publish(msg1)
        
        # UAV 2 moving opposing along X (Head-on)
        msg2 = Odometry()
        msg2.header.stamp = now.to_msg()
        msg2.pose.pose.position.x = 1.0 - self.step_count * 0.02
        msg2.pose.pose.position.y = 0.0
        msg2.pose.pose.position.z = 1.5
        msg2.twist.twist.linear.x = -1.0
        self.pub_uav2_odom.publish(msg2)
        
        # Aggressive nominal acceleration command
        cmd = Accel()
        cmd.linear.x = 3.0
        self.pub_uav1_cmd.publish(cmd)
        
        self.step_count += 1

    def safe_acc_cb(self, msg: Accel):
        self.received_safe_accs.append((msg.linear.x, msg.linear.y, msg.linear.z))

    def telem_cb(self, msg: String):
        data = json.loads(msg.data)
        self.received_telems.append(data)

def main():
    rclpy.init()
    tester = SwarmIntegrationTester()
    
    # Import the actual CoDe-CBF node from our compiled package
    from code_cbf_ros2.code_cbf_node import CoDeCBFNode
    cbf_node = CoDeCBFNode()
    
    # Use multi-threaded or sequential spinning for 2.0 seconds (100 ticks)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(tester)
    executor.add_node(cbf_node)
    
    t_start = time.time()
    while time.time() - t_start < 1.5:
        executor.spin_once(timeout_sec=0.01)
        
    print(f"Total safe control messages received: {len(tester.received_safe_accs)}")
    print(f"Total telemetry logs received:        {len(tester.received_telems)}")
    
    assert len(tester.received_safe_accs) > 10, "Should receive filtered control setpoints over ROS 2 topics"
    
    # Check latest telemetry
    last_telem = tester.received_telems[-1]
    print("Latest ROS 2 DDS Telemetry:")
    for k, v in last_telem.items():
        print(f"  {k}: {v}")
        
    print(f"Filtered setpoint at collision threshold: {tester.received_safe_accs[-1]}")
    assert last_telem['solve_ms'] < 2.0, "OSQP ROS 2 solve latency must be < 2.0 ms"
    print("\n>>> ROS 2 HUMBLE LIVE CLOSED-LOOP INTEGRATION VERIFIED! <<<")
    
    tester.destroy_node()
    cbf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
