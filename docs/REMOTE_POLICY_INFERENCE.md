# Real-robot remote policy inference

The versioned handshake, absolute-action timing and recording boundary are
documented in [the shared interface guide](ROBOT_LEARNING_CONTRACT.md).

The repository now contains a real-robot evaluation path. Ordinary teleoperation
does **not** perform model inference. A motion-armed inference launch starts the
evaluation recorder and the xArm/Robotiq reset coordinator, but not FACTR or the
teleoperation warm-up workflow. Policy motion is gated by the active evaluation
episode, so the same policy connection can run many labeled rollouts safely. A
dry-run still starts no actuator, recorder, or reset process. The local Dashboard,
mode-aware health monitor, and Rerun viewer start when `dashboard.enabled=true`.

## Closed-loop data flow

```text
ROS cameras + measured robot state       OpenPI-compatible model server
┌─────────────────────────────────┐      ┌────────────────────────────┐
│ RGB decode / crop / freshness   │      │ infer(observation)         │
│ state [8] + images + prompt     │─────▶│                            │
│ validate actions [T,8]          │◀─────│ {"actions": float32[T,8]} │
│ timed target interpolation      │      └────────────────────────────┘
└───────────────┬─────────────────┘
                │ existing ROS command topics
                ▼
        xArm follower + Robotiq controller
```

There is at most one outstanding request. A returned chunk contains waypoints on
the training dataset timeline: `policy_frequency_hz=30` matches the checked-in
30 FPS dataset. The inference node linearly interpolates those waypoints on the
xArm controller's 100 Hz monotonic clock. A 25-action chunk therefore lasts
exactly `25 / 30 = 0.833` seconds; this does not run the neural network 30 times
per second. The final waypoint occupies the last 1/30-second interval, then the
next observation is captured. Inference latency holds the previous target and
never causes catch-up bursts.

## Wire and coordinate contracts

The WebSocket endpoint is `ws://HOST:PORT`. The server first sends one binary
MessagePack metadata mapping. Every later client frame is a native OpenPI mapping:

| Field | dtype / shape | Meaning |
|---|---|---|
| `state` | `float32 [8]` | absolute `joint1..joint7` radians, then Robotiq `0..1` |
| `images.<stream_key>` | `uint8 [H,W,3]` | each selected, enabled RGB recording stream |
| `prompt` | UTF-8 string | task instruction from YAML |

Images are RGB, not ROS-driver BGR, and are sent as raw NumPy MessagePack arrays,
not JPEG. The response is `{"actions": float32[T,8]}` with `T >= 1`. Actions
are absolute robot targets in the same order and units as `state`; policy-side
normalization and image resizing beyond the configured physical training crop
belong in the model adapter.

There is no second camera/topic list under inference. `cameras.devices` and
`tactile.devices` decide which physical sensors start; `recording.streams`
defines the training feature key, ROS topic and optional center-crop/resize.
`inference.image_streams: []` automatically sends every RGB recording stream
produced by an enabled sensor. A list such as `[left_camera]` selects a subset.
A disabled or unavailable stream named explicitly is rejected during config
loading, so collection and evaluation cannot silently use different views.

ACT joint targets are published unchanged on `/xarm/absolute_target`; the xArm
controller passes those absolute radians to the SDK. FACTR alone uses the legacy
relative `/xarm/arm_pos` topic. The Robotiq topic maps leader radians with
`bits = clip(-value * 255)`, so normalized model gripper `g` is published as
`-g`. Thus model `0` is fully open and model `1` is fully closed.

## Build and start the model server

From the ROS workspace source directory:

```bash
cd /home/gxy/workspace/control_robot/src
uv sync --frozen
cd ..
source /opt/ros/humble/setup.zsh
colcon build --symlink-install --packages-select xarm_ros2_teleop
source install/setup.zsh
```

For the first test, run the included hold-position server in another terminal:

```bash
cd /home/gxy/workspace/control_robot
source /opt/ros/humble/setup.zsh
source install/setup.zsh
uv run --project src --frozen \
  python "$(command -v ros2)" run xarm_ros2_teleop remote_policy_mock_server \
  --host 0.0.0.0 --port 8000 --chunk-size 10 --delay 0.02
```

For a trained OpenPI server, keep its existing launch command and make its host,
port, observation keys and 8D absolute action adapter match the `inference:` YAML
section. The simulation reference's 16D dual-arm checkpoint is not directly
compatible with this 8D single-arm robot; it requires the correct real-robot
dataset/checkpoint or a deliberate model-side adapter.

## Mandatory no-motion dry-run

The command below starts the YAML-enabled sensors and inference node but does not
start xArm, Robotiq, FACTR, reset, or recording processes. Cameras disabled under
`cameras.devices` remain disabled. The local Dashboard and Rerun viewer remain
available for monitoring. In dry-run, absent robot feedback is replaced only for
the outbound test by the configured initial joints and an open gripper; received
actions are logged and never published.

```bash
cd /home/gxy/workspace/control_robot
source /opt/ros/humble/setup.zsh
source install/setup.zsh
uv run --project src --frozen \
  python "$(command -v ros2)" launch xarm_ros2_teleop teleop.launch.py \
  config:=src/xarm_ros2_teleop/config/teleop_system.yaml \
  inference:=true inference_motion:=false
```

Confirm all of the following before considering motion:

```bash
ros2 topic echo --once /inference/status
ros2 topic hz /camera/left/color
ros2 topic hz /camera/right/color
```

Run `ros2 topic hz` only for streams selected by the resolved YAML. The checked-in
configuration enables `top_camera`, `left_camera`, and `right_camera`; all three
must be present for the corresponding three-view ACT checkpoint.

The log must show `DRY_RUN_NO_COMMANDS`, server metadata, repeated finite 8D
actions, and no missing/stale/synchronization errors. A dry-run intentionally
cannot validate calibration, collision clearance, or model correctness.

Open <http://127.0.0.1:8080>. The inference panel must show `DRY RUN` and the
current `CONNECTING`, `INFERENCING`, or `EXECUTING_DRY_RUN` phase. The embedded
Rerun viewer displays the configured images and policy candidate actions. The
Stop button and `Esc` call `/inference/stop`; a stopped rollout cannot be restarted
inside the process and requires relaunching this command.

## Real motion

Before arming: use the hold-position mock first, clear the complete workspace,
set task-specific tighter joint limits and runtime/chunk limits in YAML, reduce
xArm speed/acceleration and Robotiq force, and keep the physical emergency stop
in hand. The launch performs the existing direct joint-space move to
`xarm.initial_joint_positions` before accepting policy commands.

Only after a successful dry-run, replace the mock with the trained server and run:

```bash
cd /home/gxy/workspace/control_robot
source /opt/ros/humble/setup.zsh
source install/setup.zsh
uv run --project src --frozen \
  python "$(command -v ros2)" launch xarm_ros2_teleop teleop.launch.py \
  config:=src/xarm_ros2_teleop/config/teleop_system.yaml \
  inference:=true inference_motion:=true
```

Stop model commands without killing ROS:

```bash
ros2 service call /inference/stop std_srvs/srv/Trigger '{}'
```

The Dashboard exposes this Stop button in dry-run or when episode evaluation is
disabled. In a labeled evaluation session it is replaced by the episode controls
below. The Web UI cannot change `inference_motion`; the page's `MOTION ARMED`
badge only reports the launch-time choice.

### Labeled evaluation session

With `evaluation.enabled=true`, the motion-armed command above opens an episode
panel at <http://127.0.0.1:8080>. First confirm the instruction, absolute dataset
directory, and Resume/Overwrite mode. The rollout controls are:

| Key | Panel action | Result |
|---|---|---|
| `S` | Start evaluation | opens recording and the policy command gate |
| `Y` | Success and save | commits the episode with reward `1.0`, then resets |
| `N` | Failure and save | commits the episode with reward `0.0`, then resets |
| `←` | Discard | removes the active attempt, then resets |
| `Esc` | Finish evaluation | discards an active unlabeled attempt and finalizes |

Each episode is limited to `evaluation.max_episode_sec` (30 seconds by default).
A timeout is saved as a truncated failure. A nonzero xArm controller error or
loss of control health is saved as a terminal failure; the current policy gate
closes immediately and the recorder waits for automatic xArm/Robotiq recovery
before another rollout can start. Evaluation reset deliberately has no FACTR
dependency.

LeRobot observations, states, and executed actions remain in the selected
dataset. Auditable labels are appended to
`<dataset_root>/evaluation/outcomes.jsonl`; each line records the zero-based
episode index, outcome, scalar reward, terminal/truncated flags, reason, duration,
frame count, FPS, instruction, and timestamp. Discarded attempts do not receive
an episode index and are not added to the LeRobot dataset.

`Ctrl-C` also stops the rollout. The stop control publishes the latest measured
pose once, then sends no further policy targets; the normal xArm controller holds
that target. It does not enter the SDK emergency-stop state, so coordinated reset
remains available. For unexpected motion or any doubt, use the physical emergency
stop immediately.

## Safety guards and termination

Before a target is published, the node requires:

- fresh robot state and all configured images;
- camera arrival-time skew within `max_camera_skew_sec`;
- healthy `/xarm/control_ok` and inactive reset;
- action shape `[T,8]`, finite values, gripper in `[0,1]`;
- every joint within configured absolute limits;
- the first waypoint differs from the measured state submitted to ACT by no
  more than `max_joint_step_rad`, and every later waypoint obeys the same large
  emergency bound. The default real-robot limit is `0.5 rad`; ordinary tracking
  lag and chunk-boundary variation are intentionally allowed.

Outside episode-gated evaluation, the rollout stops on an inference timeout, a
health transition from good to bad, reset activation, invalid server output,
`max_runtime_sec`, `max_chunks`, the stop service, process interruption, or a
critical node exit. During evaluation, timeout, chunk-limit and invalid-response
failures terminate only the active episode: the recorder labels it, performs the
normal reset, and reuses the same policy connection for the next episode. Network
requests don't block ROS callbacks.
