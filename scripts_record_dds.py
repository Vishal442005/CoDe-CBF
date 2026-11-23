
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import time, csv

rclpy.init()
node = Node('latency_recorder')
pub = node.create_publisher(Header, 'uav_odom_test', 10)

records = []
def listener_cb(msg):
    recv_time = time.time()
    sent_time = msg.stamp.sec + msg.stamp.nanosec * 1e-9
    latency_ms = (recv_time - sent_time) * 1000.0
    records.append((len(records)+1, sent_time, recv_time, latency_ms))

sub = node.create_subscription(Header, 'uav_odom_test', listener_cb, 10)

# Run 100 packets at 50 Hz
t_start = time.time()
target_packets = 100
interval = 0.02

for i in range(target_packets):
    now = time.time()
    msg = Header()
    msg.stamp.sec = int(now)
    msg.stamp.nanosec = int((now - int(now)) * 1e9)
    pub.publish(msg)
    
    # Process callbacks
    t_end = now + interval
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.002)

with open('/home/jarvis/Drones_CoDe_CBF/data/raw_network/real_ros2_dds_latency_trace.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['packet_seq', 'sent_timestamp', 'recv_timestamp', 'latency_ms'])
    writer.writerows(records)

print(f"Recorded {len(records)} packets. Avg latency: {sum(r[3] for r in records)/max(1, len(records)):.3f} ms")
node.destroy_node()
rclpy.shutdown()
