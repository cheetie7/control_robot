"""Validate the checked-in unified system configuration."""

from pathlib import Path

import pytest
import yaml
from xarm_ros2_teleop.system.config import load_system_config


def test_default_config_is_valid_and_sensors_are_enabled() -> None:
    """Checked-in hardware serials must be active for the full teleop launch."""

    config_path = Path(__file__).parents[1] / "config" / "teleop_system.yaml"
    system = load_system_config(config_path)
    calibration_pose = [0.0, 0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
    assert system.teleop["enabled"] is True
    assert system.teleop["offset_file"].endswith("offset.yaml")
    assert system.teleop["xarm_to_factr_signs"] == [1] * 7
    assert system.xarm["enabled"] is True
    assert system.xarm["state_read_rate_hz"] == 60.0
    assert system.xarm["max_joint_speed_rad_s"] == 1.2
    assert system.xarm["max_joint_acceleration_rad_s2"] == 1.28
    assert {"joints", "is_radian", "arm_name"}.isdisjoint(system.xarm)
    assert system.robotiq["enabled"] is True
    assert system.robotiq["com_port"] == "/dev/robotiq_rs485"
    assert system.robotiq["activation_timeout_sec"] == 15.0
    assert system.inference.enabled is False
    assert system.inference.dry_run is True
    assert system.inference.policy_frequency_hz == 30.0
    assert system.evaluation["enabled"] is True
    assert system.evaluation["dataset_root"] == "/home/gxy/datasets/eval"
    assert system.evaluation["max_episode_sec"] == 30.0
    assert system.evaluation["success_reward"] == 1.0
    assert system.evaluation["failure_reward"] == 0.0
    assert system.rl["enabled"] is False
    assert system.rl["stop_velocity_rad_s"] == 0.03
    assert system.rl["success_reward"] == 1.0
    # Inference automatically shares all three enabled RGB recording streams
    # and their exact topic/crop definitions.
    assert [image.key for image in system.inference.images] == [
        "top_camera",
        "left_camera",
        "right_camera",
    ]
    assert system.reset["enabled"] is True
    assert system.xarm["initial_joint_positions"] == calibration_pose
    assert system.cameras_enabled is True
    assert len(system.cameras) == 3
    assert sum(camera.enabled for camera in system.cameras) == 3
    # This collection profile uses only three RGB cameras.
    assert system.tactile_enabled is False
    assert len(system.tactile_devices) == 2
    assert system.recording.enabled is True
    assert [stream.key for stream in system.recording.streams] == [
        "top_camera",
        "left_camera",
        "right_camera",
    ]
    assert system.recording.instruction == "teleoperate the robot to complete the task"
    assert system.recording.dataset_name == "lerobot_dataset"
    assert system.recording.append_timestamp is True
    assert system.recording.num_episodes == 1
    assert system.recording.reset_joint_positions == tuple(system.xarm["initial_joint_positions"])
    assert system.recording.max_state_skew_sec == 0.075
    assert system.recording.max_action_age_sec == 0.075
    assert system.recording.action_history_sec == 1.0
    assert system.recording.writer_queue_size == 16
    assert system.warmup.enabled is True
    assert system.warmup.finished_delay_sec == 2.0
    assert system.speech.phrase("warmup_finished") == "warm up finished"
    assert system.dashboard.enabled is True
    assert system.dashboard.host == "127.0.0.1"
    assert system.dashboard.rerun.grpc_port == 9876


def test_five_sensor_profile_cannot_start_actuators() -> None:
    """The sensor test profile must enable five sensors and no motion process."""

    package_root = Path(__file__).parents[1]
    system = load_system_config(package_root / "config" / "sensors_all.yaml")
    assert system.teleop["enabled"] is False
    assert system.xarm["enabled"] is False
    assert system.robotiq["enabled"] is False
    assert system.reset["enabled"] is False
    assert system.recording.enabled is False
    assert system.inference.enabled is False
    assert system.rl["enabled"] is False
    assert system.warmup.enabled is False
    assert system.cameras_enabled is True
    assert system.tactile_enabled is True
    assert sum(camera.enabled for camera in system.cameras) == 3
    assert sum(device.enabled for device in system.tactile_devices) == 2
    assert len(system.recording.streams) == 5
    assert system.dashboard.enabled is True
    assert system.dashboard.rerun.enabled is True


def test_workspace_public_launches_and_two_ros_packages() -> None:
    """Keep camera and bringup code inside xarm_ros2_teleop rather than extra packages."""

    source_root = Path(__file__).parents[2]
    package_manifests = sorted(path.parent.name for path in source_root.glob("*/package.xml"))
    public_launches = list(source_root.glob("*/launch/*.launch.py"))
    assert package_manifests == ["factr_teleop", "xarm_ros2_teleop"]
    assert sorted(path.name for path in public_launches) == [
        "quest_pose.launch.py",
        "sensors.launch.py",
        "teleop.launch.py",
    ]
    sensor_launch = source_root / "xarm_ros2_teleop" / "launch" / "sensors.launch.py"
    sensor_source = sensor_launch.read_text(encoding="utf-8")
    assert all(
        executable not in sensor_source
        for executable in ("factr_teleop_node", "xarm_node", "robotiq_node", "reset_node")
    )
    generic_nodes = list((source_root / "xarm_ros2_teleop" / "xarm_ros2_teleop").glob("*/node.py"))
    assert generic_nodes == []
    compatibility_modules = (
        "factr_teleop.py",
        "factr_teleop_ros2.py",
        "xarm_node.py",
    )
    package_root = source_root / "xarm_ros2_teleop" / "xarm_ros2_teleop"
    assert all(not (package_root / module).exists() for module in compatibility_modules)
    assert not (source_root / "xarm_ros2_teleop" / "franka_example.yaml").exists()


def test_factr_config_contains_only_effective_dynamixel_fields() -> None:
    """Removed experimental fields must not imply nonexistent current behavior."""

    config_path = Path(__file__).parents[1] / "config" / "gello_grav_comp_demo.yaml"
    dynamixel = yaml.safe_load(config_path.read_text(encoding="utf-8"))["dynamixel"]
    removed_fields = {
        "current_limit_ma",
        "current_signs",
        "controlled_ids",
        "allow_arbitrary_offsets",
    }
    assert removed_fields.isdisjoint(dynamixel)
    assert "arm_joint_offsets_rad" not in dynamixel
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["arm_teleop"]["initialization"][
        "start_joints"
    ] == [0.0, 0.0, 0.0, 1.57, 0.0, 1.57, 0.0, 0.0]


def test_reset_requires_all_three_actuator_components(tmp_path: Path) -> None:
    """Enabled reset may not start without the independently connected gripper."""

    source = Path(__file__).parents[1] / "config" / "teleop_system.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["robotiq"]["enabled"] = False
    target = tmp_path / "missing_gripper.yaml"
    target.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="requires teleop, xarm and robotiq"):
        load_system_config(target)


def test_reset_rejects_invalid_initial_joint_array(tmp_path: Path) -> None:
    """The shared startup and fallback pose must always contain seven radians."""

    source = Path(__file__).parents[1] / "config" / "teleop_system.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["xarm"]["initial_joint_positions"] = [0.0] * 6
    target = tmp_path / "invalid_initial.yaml"
    target.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain 7 values"):
        load_system_config(target)


def test_inference_reuses_enabled_recording_rgb_streams(tmp_path: Path) -> None:
    """Policy image selection must share sensor topics and crops with collection."""

    source = Path(__file__).parents[1] / "config" / "teleop_system.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["inference"]["image_streams"] = ["left_camera"]
    target = tmp_path / "left_camera_policy.yaml"
    target.write_text(yaml.safe_dump(config), encoding="utf-8")
    system = load_system_config(target)
    assert [(image.key, image.topic) for image in system.inference.images] == [
        ("left_camera", "/camera/left/color")
    ]

    config["inference"]["image_streams"] = ["top_depth"]
    target.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="not enabled RGB sensor streams"):
        load_system_config(target)


def test_recording_accepts_only_streams_generated_by_enabled_sensors(tmp_path: Path) -> None:
    """Recorder startup validates its complete fixed video schema before hardware opens."""

    source = Path(__file__).parents[1] / "config" / "teleop_system.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["cameras"]["enabled"] = True
    # The recording anchor (top_camera) requires the Femto Bolt color topic.
    config["cameras"]["devices"][2]["enabled"] = True
    config["tactile"]["enabled"] = True
    config["recording"]["enabled"] = True
    valid_path = tmp_path / "recording_valid.yaml"
    valid_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    system = load_system_config(valid_path)
    assert system.recording.anchor_key == "top_camera"
    # This profile explicitly records only the three RGB views.
    assert len(system.recording.streams) == 3

    # Streams whose sensor is disabled are auto-pruned with a warning instead
    # of failing the whole configuration.
    config["tactile"]["devices"][0]["outputs"]["rectify"] = True
    config["tactile"]["devices"][0]["enabled"] = False
    config["tactile"]["enabled"] = True
    config["recording"]["streams"]["tactile_left"]["enabled"] = True
    config["recording"]["streams"]["tactile_right"]["enabled"] = True
    pruned_path = tmp_path / "recording_pruned.yaml"
    pruned_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    system = load_system_config(pruned_path)
    pruned_keys = [stream.key for stream in system.recording.streams]
    # Left tactile sensor disabled -> its stream pruned; right stays.
    assert "tactile_left" not in pruned_keys
    assert "tactile_right" in pruned_keys
    assert "top_camera" in pruned_keys


def test_post_episode_reset_requires_reset_coordinator(tmp_path: Path) -> None:
    """A configured return pose cannot be used without the coordinated reset node."""

    source = Path(__file__).parents[1] / "config" / "teleop_system.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["cameras"]["enabled"] = True
    # The recording anchor (top_camera) requires the Femto Bolt color topic.
    config["cameras"]["devices"][2]["enabled"] = True
    config["tactile"]["enabled"] = True
    config["recording"]["enabled"] = True
    config["reset"]["enabled"] = False
    target = tmp_path / "recording_without_reset.yaml"
    target.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="episode recording reset requires reset.enabled"):
        load_system_config(target)
