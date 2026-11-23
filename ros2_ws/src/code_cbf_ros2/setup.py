from setuptools import setup
import os
from glob import glob

package_name = 'code_cbf_ros2'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jarvis',
    maintainer_email='jarvis@antigravity.ai',
    description='CoDe-CBF Decentralized Safety Filter Node for ROS 2 Humble',
    license='MIT',
    entry_points={
        'console_scripts': [
            'code_cbf_node = code_cbf_ros2.code_cbf_node:main',
        ],
    },
)
