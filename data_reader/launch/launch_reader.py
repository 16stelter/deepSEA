from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('data_reader'),
        'config',
        'sensors.yaml'
    )
    return LaunchDescription([
        Node(
            package='data_reader',
            executable='reader',
            name='reader',
            output='screen',
            emulate_tty=True,
            parameters=[
                config
            ]
        ),
    ])
