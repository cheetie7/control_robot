# Meta Quest controller pose bridge

Schema 3 adds coherent grip/trigger inputs, input validity, session identity and
tracking-origin revision. Schema 2 remains available for pose diagnostics only.
For the integrated native-TCP recording route, see [VR 遥操作说明](VR_TELEOP_ZH.md).
Rebuild the APK after this protocol upgrade; ROS rebuild alone does not update it.

The bridge publishes Meta Quest left and right controller poses relative to the
Floor Level tracking origin as ROS 2 messages. The Quest application streams at approximately 60 Hz
through a TCP connection carried over the USB debugging link, so the headset and
Ubuntu computer do not need to share a Wi-Fi network.

## Run

Connect the Quest over USB, then run from the workspace `src/` directory:

```bash
./scripts/diagnostics/run_quest_pose.sh --build
```

After the first build, omit `--build`:

```bash
./scripts/diagnostics/run_quest_pose.sh
```

Open the bridge and a minimal RViz pose view together:

```bash
./scripts/diagnostics/run_quest_pose.sh --rviz
```

The program waits up to 120 seconds for the only authorized Quest, wakes it, launches
`com.controlrobot.questposebridge`, and creates the USB reverse-port mapping. If
several devices are connected:

```bash
./scripts/diagnostics/run_quest_pose.sh --device 3487C10J6W00PZ
```

Developer mode and USB debugging authorization are one-time prerequisites. Resuming a
still-running pose app normally requires no confirmation inside the headset. After a
full headset reboot, a stopped app process, some system updates, or an invalid saved
boundary, Meta Horizon OS can require Guardian confirmation before starting the XR
session; an application cannot bypass that system safety dialog.

Horizon OS can temporarily remove the USB mapping while showing a boundary or other
system dialog. The bridge monitors the pose stream and automatically restores that
mapping after the dialog closes. If the system returns to Quest Home instead of the
pose app, the bridge also brings the app back to the foreground automatically; no
additional terminal command is required.

The controller pose topics become valid after the headset is worn and the controllers
are inside the tracking-camera field of view. Until then, the corresponding `tracked`
topics remain false.

## Topics

| Topic | Type | Meaning |
|---|---|---|
| `/quest/left_controller/pose` | `geometry_msgs/msg/PoseStamped` | Valid left controller pose relative to the floor origin |
| `/quest/right_controller/pose` | `geometry_msgs/msg/PoseStamped` | Valid right controller pose relative to the floor origin |
| `/quest/visualization/markers` | `visualization_msgs/msg/MarkerArray` | Global axes, current controller axes, and position dots from the previous 3 seconds (10 Hz) |
| `/quest/left_controller/tracked` | `std_msgs/msg/Bool` | Whether the current left pose is valid |
| `/quest/right_controller/tracked` | `std_msgs/msg/Bool` | Whether the current right pose is valid |
| `/quest/bridge/status` | `std_msgs/msg/String` | Connected, disconnected, and stale-stream state |

Inspect one stream in another sourced terminal:

```bash
ros2 topic echo /quest/right_controller/pose
ros2 topic hz /quest/right_controller/pose
```

`PoseStamped.header.frame_id` is `quest_floor`. Coordinates follow ROS FLU:

- `+x`: forward along the tracking origin's horizontal reference direction
- `+y`: left
- `+z`: up
- position unit: metres
- rotation: xyzw quaternion

The reference frame follows the Quest runtime's Floor Level tracking origin, not
the headset's current tilt or position. Floor height comes from the saved boundary
setup. Unity `+y` becomes ROS `+z` with scale 1: moving a controller vertically by
0.20 m changes its ROS height by 0.20 m, even when the headset looks down. RViz uses
`quest_floor` as its fixed frame, draws an XY floor grid at `z=0`, and retains the
current axes and only the last three seconds of position dots.

The Quest application requests Floor Level explicitly and marks samples invalid
until the XR subsystem reports a running floor origin. Version-2 network frames
include `frameId: quest_floor`; the receiver rejects old head-relative APK data
instead of relabeling it. Install the updated APK when migrating from `quest_head`.

Resetting the view or changing the boundary can change the origin and horizontal
heading. This is a local tracking frame, not a persistent robot-base calibration.
For later 1:1 teleoperation, record the controller and robot start poses when
engaging control, and re-engage after an origin reset or tracking loss.

Reference: [Meta XR camera settings](https://developers.meta.com/vr/documentation/unity/unity-ovrcamerarig/).

The pose topic is not updated while tracking is invalid. Consumers controlling a
robot must use the corresponding `tracked` topic as a safety gate and stop motion
when it is false. The bridge forces both tracking topics to false when no new frame
arrives for 0.25 seconds.

## Rebuild the Quest application

The Unity source is in `quest_pose_bridge/`. After changing it, rebuild
`Builds/QuestPoseBridge.apk` through `QuestPoseBridgeSetup.BuildAndroid`, then install:

```bash
metavr app install quest_pose_bridge/Builds/QuestPoseBridge.apk
```

Normal ROS-side runs do not require reinstalling the APK.
