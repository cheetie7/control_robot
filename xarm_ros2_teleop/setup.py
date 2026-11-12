from glob import glob

from setuptools import find_packages, setup

package_name = "xarm_ros2_teleop"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/urdf", glob("urdf/*.urdf")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=[
        "fastapi>=0.115,<1",
        "lerobot==0.4.4",
        "msgpack>=1.0,<2",
        "pyrobotiqgripper==3.2.8",
        "rerun-sdk==0.26.2",
        "setuptools",
        "uvicorn[standard]>=0.30,<1",
        "websockets>=15,<16",
    ],
    package_data={"xarm_ros2_teleop.dashboard": ["static/*"]},
    include_package_data=True,
    zip_safe=True,
    maintainer="root",
    maintainer_email="1134060136@sjtu.edu.cn",
    description="Unified FACTR-to-xArm teleoperation with RGB-D and Xense sensors",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "xarm_node = xarm_ros2_teleop.xarm.xarm_controller:main",
            "factr_teleop_node = xarm_ros2_teleop.teleop.leader_node:main",
            "factr_calibrate = xarm_ros2_teleop.teleop.factr_calibration_cli:main",
            "factr_reset_debug = xarm_ros2_teleop.teleop.factr_reset_debug:main",
            "robotiq_node = xarm_ros2_teleop.robotiq.robotiq_controller:main",
            "reset_node = xarm_ros2_teleop.reset.reset_coordinator:main",
            "tactile_node = xarm_ros2_teleop.tactile.xense_publisher:main",
            "episode_recorder = xarm_ros2_teleop.recording.episode_recorder:main",
            "rl_episode_recorder = xarm_ros2_teleop.rl.episode_recorder:main",
            "system_monitor = xarm_ros2_teleop.monitor.system_monitor:main",
            "rerun_visualizer = xarm_ros2_teleop.visualization.rerun_visualizer:main",
            "dashboard_server = xarm_ros2_teleop.dashboard.server:main",
            "session_coordinator = xarm_ros2_teleop.session.session_coordinator:main",
            "speech_announcer = xarm_ros2_teleop.speech.speech_announcer:main",
            "remote_policy_inference = xarm_ros2_teleop.inference.remote_policy_node:main",
            "remote_policy_mock_server = xarm_ros2_teleop.inference.mock_server:main",
            "human_intervention = xarm_ros2_teleop.rl.human_intervention_node:main",
            "quest_pose_node = xarm_ros2_teleop.vr.quest_pose_node:main",
            "vr_teleop = xarm_ros2_teleop.vr.teleop_node:main",
        ],
    },
)
