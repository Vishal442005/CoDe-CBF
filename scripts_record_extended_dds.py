
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import time, csv, random

rclpy.init()
node = Node('latency_recorder_extended')
pub = node.create_publisher(Header, 'uav_odom_stream', 20)

records = []
received_seqs = set()
latest_received_tick = -1

# Markov 2-state channel for burst loss: Good (drop 2%) vs Bad (drop 40%)
state = 'G' # 'G' or 'B'
p_gb = 0.05
p_bg = 0.20

def listener_cb(msg):
    global latest_received_tick
    recv_time = time.time()
    seq = msg.stamp.sec
    sent_time = msg.stamp.nanosec * 1e-9 + (msg.stamp.sec * 0.02)
    latency_ms = (recv_time - (t_zero + seq * 0.02)) * 1000.0
    records.append((seq, latency_ms, True))
    received_seqs.add(seq)
    latest_received_tick = max(latest_received_tick, seq)

sub = node.create_subscription(Header, 'uav_odom_stream', listener_cb, 20)

target_packets = 1000
interval = 0.005 # fast stream for 1000 samples
t_zero = time.time()

for seq in range(target_packets):
    now = time.time()
    # Markov channel transition
    if state == 'G':
        if random.random() < p_gb: state = 'B'
    else:
        if random.random() < p_bg: state = 'G'
    
    # Drop probability based on real wireless fading channel
    drop_prob = 0.03 if state == 'G' else 0.45
    will_drop = random.random() < drop_prob
    
    if not will_drop:
        msg = Header()
        msg.stamp.sec = seq
        msg.stamp.nanosec = int((now - int(now)) * 1e9)
        pub.publish(msg)
    else:
        # dropped packet
        records.append((seq, -1.0, False))

    t_end = now + interval
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.001)

# Compute Age-of-Information (AoI)
final_rows = []
current_last_recv = 0
for seq in range(target_packets):
    # Find record for this seq
    rec = [r for r in records if r[0] == seq]
    if rec and rec[0][2]: # successfully received
        current_last_recv = seq
        lat = rec[0][1]
        dropped = 0
    else:
        lat = -1.0
        dropped = 1
    aoi_ticks = seq - current_last_recv
    aoi_ms = aoi_ticks * 20.0 # 50 Hz control tick
    final_rows.append((seq, state, dropped, lat, aoi_ticks, aoi_ms))

with open('/home/jarvis/Drones_CoDe_CBF/data/raw_network/ros2_dds_extended_trace.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seq', 'channel_state', 'is_dropped', 'latency_ms', 'aoi_ticks', 'aoi_ms'])
    writer.writerows(final_rows)

print(f"Recorded {len(final_rows)} packets. Total drops: {sum(r[2] for r in final_rows)} ({sum(r[2] for r in final_rows)/len(final_rows)*100:.1f}%)")
print(f"Max AoI: {max(r[4] for r in final_rows)} ticks ({max(r[5] for r in final_rows):.1f} ms)")
node.destroy_node()
rclpy.shutdown()
