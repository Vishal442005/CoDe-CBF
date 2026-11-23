import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
    
    total_uavs = 4
    r_safe = 0.8
    kappa_z = 2.0
    dt = 0.02
    
    # Launch 4 decentralized CoDe-CBF safety filter nodes
    for uav_id in range(1, total_uavs + 1):
        node = Node(
            package='code_cbf_ros2',
            executable='code_cbf_node',
            name=f'cbf_filter_uav_{uav_id}',
            parameters=[{
                'uav_id': uav_id,
                'total_uavs': total_uavs,
                'r_safe': r_safe,
                'kappa_z': kappa_z,
                'dt': dt,
                'alpha_target': 0.05,
                'lambda_aoi': 1.5
            }],
            output='screen'
        )
        ld.add_action(node)
        
    return ld
