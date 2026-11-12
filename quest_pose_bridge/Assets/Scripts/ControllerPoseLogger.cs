using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.XR;

namespace ControlRobot.QuestPose
{
    /// <summary>
    /// Samples controller poses in the floor-level tracking space and streams newline-delimited JSON.
    /// Position values are Unity metres and rotations are xyzw unit quaternions.
    /// </summary>
    public sealed class ControllerPoseLogger : MonoBehaviour
    {
        [SerializeField] private Transform trackingSpace;
        [SerializeField] private Transform leftController;
        [SerializeField] private Transform rightController;
        [SerializeField, Min(1f)] private float streamRateHz = 60f;
        [SerializeField, Min(0.1f)] private float diagnosticLogRateHz = 2f;
        [SerializeField] private string bridgeHost = "127.0.0.1";
        [SerializeField, Range(1, 65535)] private int bridgePort = 7447;

        private double nextStreamTimeSeconds;
        private double nextDiagnosticLogTimeSeconds;
        private LatestFrameTcpSender tcpSender;
        private readonly string sessionId = Guid.NewGuid().ToString("N");
        private int originRevision;
        private readonly HashSet<XRInputSubsystem> observedSubsystems = new HashSet<XRInputSubsystem>();
        private readonly List<XRInputSubsystem> inputSubsystems = new List<XRInputSubsystem>();

        /// <summary>
        /// Assigns the tracked transforms used by the pose sampler.
        /// </summary>
        /// <param name="trackingSpaceTransform">OVRCameraRig tracking space with a floor-level origin.</param>
        /// <param name="leftControllerTransform">Left controller transform in Unity world coordinates.</param>
        /// <param name="rightControllerTransform">Right controller transform in Unity world coordinates.</param>
        /// <returns>No value.</returns>
        public void Configure(
            Transform trackingSpaceTransform,
            Transform leftControllerTransform,
            Transform rightControllerTransform)
        {
            trackingSpace = trackingSpaceTransform;
            leftController = leftControllerTransform;
            rightController = rightControllerTransform;
        }

        /// <summary>
        /// Validates pose sources and starts the non-blocking TCP sender.
        /// </summary>
        /// <returns>No value.</returns>
        private void Awake()
        {
            if (trackingSpace == null || leftController == null || rightController == null)
            {
                Debug.LogError("QUEST_POSE missing tracking space or controller transforms; disabling streamer.");
                enabled = false;
                return;
            }

            if (streamRateHz <= 0f || diagnosticLogRateHz <= 0f || bridgePort <= 0)
            {
                Debug.LogError("QUEST_POSE invalid stream configuration; disabling streamer.");
                enabled = false;
                return;
            }

            tcpSender = new LatestFrameTcpSender(bridgeHost, bridgePort);
            tcpSender.Start();
        }

        /// <summary>
        /// Samples a coherent two-controller frame and queues only the newest frame for delivery.
        /// </summary>
        /// <returns>No value.</returns>
        private void LateUpdate()
        {
            double nowSeconds = Time.realtimeSinceStartupAsDouble;
            if (nowSeconds < nextStreamTimeSeconds)
            {
                return;
            }

            double streamIntervalSeconds = 1.0 / streamRateHz;
            nextStreamTimeSeconds += streamIntervalSeconds;
            if (nextStreamTimeSeconds <= nowSeconds - streamIntervalSeconds)
            {
                // Drop missed samples after a long pause without reducing the steady-state rate.
                nextStreamTimeSeconds = nowSeconds + streamIntervalSeconds;
            }
            // Sample after OVRCameraRig.Update; do not label eye-level poses as floor-level poses.
            SubsystemManager.GetSubsystems(inputSubsystems);
            foreach (XRInputSubsystem subsystem in inputSubsystems)
            {
                if (observedSubsystems.Add(subsystem))
                {
                    subsystem.trackingOriginUpdated += OnTrackingOriginUpdated;
                }
            }
            bool floorTrackingReady = inputSubsystems.Exists(subsystem =>
                subsystem.running && subsystem.GetTrackingOriginMode() == TrackingOriginModeFlags.Floor);
            PoseSample leftSample = CreateSample("left", XRNode.LeftHand, leftController, floorTrackingReady);
            PoseSample rightSample = CreateSample("right", XRNode.RightHand, rightController, floorTrackingReady);
            var frame = new PoseFrame(nowSeconds, leftSample, rightSample, sessionId, originRevision);
            tcpSender.PublishLatest(JsonUtility.ToJson(frame));

            if (nowSeconds >= nextDiagnosticLogTimeSeconds)
            {
                nextDiagnosticLogTimeSeconds = nowSeconds + 1.0 / diagnosticLogRateHz;
                Debug.Log($"QUEST_POSE floorTrackingReady={floorTrackingReady}");
                Debug.Log($"QUEST_POSE {JsonUtility.ToJson(leftSample)}");
                Debug.Log($"QUEST_POSE {JsonUtility.ToJson(rightSample)}");
            }
        }

        /// <summary>
        /// Stops the network worker when Unity destroys the component.
        /// </summary>
        /// <returns>No value.</returns>
        private void OnDestroy()
        {
            foreach (XRInputSubsystem subsystem in observedSubsystems)
            {
                subsystem.trackingOriginUpdated -= OnTrackingOriginUpdated;
            }
            tcpSender?.Dispose();
        }

        /// <summary>Invalidate control origins after runtime recenter or origin changes.</summary>
        /// <param name="subsystem">XR subsystem reporting the changed tracking origin.</param>
        /// <returns>No value; the next network frame carries the new revision.</returns>
        private void OnTrackingOriginUpdated(XRInputSubsystem subsystem)
        {
            originRevision++;
        }

        /// <summary>
        /// Creates one tracked or explicitly invalid controller sample.
        /// </summary>
        /// <param name="hand">Logical controller side: left or right.</param>
        /// <param name="node">Unity XR node used to check tracking validity.</param>
        /// <param name="controller">Controller transform in Unity world coordinates.</param>
        /// <param name="floorTrackingReady">True while the XR runtime uses a running floor origin.</param>
        /// <returns>Controller pose expressed in floor tracking coordinates, or an invalid sample.</returns>
        private PoseSample CreateSample(string hand, XRNode node, Transform controller, bool floorTrackingReady)
        {
            bool tracked = floorTrackingReady && IsTracked(node);
            Pose relativePose = tracked
                ? GetRelativePose(trackingSpace, controller)
                : new Pose(Vector3.zero, Quaternion.identity);
            InputDevice device = InputDevices.GetDeviceAtXRNode(node);
            bool grip = false;
            float trigger = 0f;
            bool inputValid = device.isValid
                && device.TryGetFeatureValue(CommonUsages.gripButton, out grip)
                && device.TryGetFeatureValue(CommonUsages.trigger, out trigger);
            return new PoseSample(hand, tracked, relativePose.position, relativePose.rotation,
                inputValid, grip, Mathf.Clamp01(trigger));
        }

        /// <summary>
        /// Checks whether one XR controller currently provides a valid tracked pose.
        /// </summary>
        /// <param name="node">LeftHand or RightHand XR node.</param>
        /// <returns>True when the XR runtime reports the controller as tracked.</returns>
        private static bool IsTracked(XRNode node)
        {
            InputDevice device = InputDevices.GetDeviceAtXRNode(node);
            return device.isValid
                && device.TryGetFeatureValue(CommonUsages.isTracked, out bool tracked)
                && tracked
                && device.TryGetFeatureValue(CommonUsages.trackingState, out InputTrackingState state)
                && (state & (InputTrackingState.Position | InputTrackingState.Rotation))
                    == (InputTrackingState.Position | InputTrackingState.Rotation);
        }

        /// <summary>
        /// Converts a controller world pose into the gravity-aligned floor tracking frame.
        /// </summary>
        /// <param name="referenceTransform">Floor tracking origin; its scale must be one.</param>
        /// <param name="controllerTransform">Controller transform in the same world frame.</param>
        /// <returns>Pose in metres and xyzw quaternion order relative to the floor origin.</returns>
        private static Pose GetRelativePose(Transform referenceTransform, Transform controllerTransform)
        {
            // T_floor_controller = inverse(T_world_floor) * T_world_controller.
            // The center-eye transform is not part of this calculation, so head tilt has no effect.
            Vector3 positionMeters = referenceTransform.InverseTransformPoint(controllerTransform.position);
            Quaternion rotation = Quaternion.Inverse(referenceTransform.rotation) * controllerTransform.rotation;
            return new Pose(positionMeters, rotation);
        }

        /// <summary>
        /// Serializable wire record containing both controllers at one Unity timestamp.
        /// </summary>
        [Serializable]
        private struct PoseFrame
        {
            public int schemaVersion;
            public string frameId;
            public double timestampSeconds;
            public string sessionId;
            public int originRevision;
            public PoseSample left;
            public PoseSample right;

            /// <summary>
            /// Creates one coherent two-controller network frame.
            /// </summary>
            /// <param name="timestampSecondsValue">Monotonic Unity time in seconds.</param>
            /// <param name="leftValue">Left controller sample.</param>
            /// <param name="rightValue">Right controller sample.</param>
            /// <param name="sessionIdValue">Unique identity of this application run.</param>
            /// <param name="originRevisionValue">Monotonic tracking-origin revision.</param>
            public PoseFrame(
                double timestampSecondsValue,
                PoseSample leftValue,
                PoseSample rightValue,
                string sessionIdValue,
                int originRevisionValue)
            {
                schemaVersion = 3;
                frameId = "quest_floor";
                timestampSeconds = timestampSecondsValue;
                left = leftValue;
                right = rightValue;
                sessionId = sessionIdValue;
                originRevision = originRevisionValue;
            }
        }

        /// <summary>
        /// Serializable controller pose in the floor-relative Unity coordinate frame.
        /// </summary>
        [Serializable]
        private struct PoseSample
        {
            public string hand;
            public bool tracked;
            public Vector3 positionMeters;
            public Quaternion rotationXyzw;
            public bool inputValid;
            public bool grip;
            public float trigger;

            /// <summary>
            /// Creates one controller sample ready for JSON serialization.
            /// </summary>
            /// <param name="handValue">Logical controller side: left or right.</param>
            /// <param name="trackedValue">Whether the runtime reports a valid pose.</param>
            /// <param name="positionMetersValue">Three-element position in metres.</param>
            /// <param name="rotationXyzwValue">Unit quaternion in xyzw order.</param>
            /// <param name="inputValidValue">True only if grip and trigger were both read.</param>
            /// <param name="gripValue">Current physical grip button state.</param>
            /// <param name="triggerValue">Trigger closure fraction in [0,1].</param>
            public PoseSample(
                string handValue,
                bool trackedValue,
                Vector3 positionMetersValue,
                Quaternion rotationXyzwValue,
                bool inputValidValue,
                bool gripValue,
                float triggerValue)
            {
                hand = handValue;
                tracked = trackedValue;
                positionMeters = positionMetersValue;
                rotationXyzw = rotationXyzwValue;
                inputValid = inputValidValue;
                grip = gripValue;
                trigger = triggerValue;
            }
        }

        /// <summary>
        /// Sends only the latest JSON frame on a background TCP worker.
        /// The device connects to localhost; metavr reverse maps that port to Ubuntu over USB.
        /// </summary>
        private sealed class LatestFrameTcpSender : IDisposable
        {
            private readonly string host;
            private readonly int port;
            private readonly object payloadLock = new object();
            private readonly AutoResetEvent frameReady = new AutoResetEvent(false);

            private Thread worker;
            private string latestPayload;
            private volatile bool running;

            /// <summary>
            /// Creates a sender targeting one TCP endpoint.
            /// </summary>
            /// <param name="hostValue">IPv4 host or DNS name visible from the Quest process.</param>
            /// <param name="portValue">TCP port in the range 1-65535.</param>
            public LatestFrameTcpSender(string hostValue, int portValue)
            {
                host = hostValue;
                port = portValue;
            }

            /// <summary>
            /// Starts the background connection and send loop once.
            /// </summary>
            /// <returns>No value.</returns>
            public void Start()
            {
                if (running)
                {
                    return;
                }

                running = true;
                worker = new Thread(SendLoop)
                {
                    IsBackground = true,
                    Name = "QuestPoseTcpSender"
                };
                worker.Start();
            }

            /// <summary>
            /// Replaces any unsent frame so network stalls never accumulate control latency.
            /// </summary>
            /// <param name="json">One compact JSON object without a trailing newline.</param>
            /// <returns>No value.</returns>
            public void PublishLatest(string json)
            {
                lock (payloadLock)
                {
                    latestPayload = json;
                }

                frameReady.Set();
            }

            /// <summary>
            /// Connects, writes newline-delimited frames, and reconnects after transport errors.
            /// </summary>
            /// <returns>No value.</returns>
            private void SendLoop()
            {
                TcpClient client = null;
                NetworkStream stream = null;
                while (running)
                {
                    frameReady.WaitOne(100);
                    string payload = TakeLatestPayload();
                    if (!running || payload == null)
                    {
                        continue;
                    }

                    try
                    {
                        if (client == null || !client.Connected)
                        {
                            CloseConnection(ref stream, ref client);
                            client = Connect();
                            stream = client.GetStream();
                        }

                        byte[] bytes = Encoding.UTF8.GetBytes(payload + "\n");
                        stream.Write(bytes, 0, bytes.Length);
                    }
                    catch (Exception exception) when (
                        exception is SocketException
                        || exception is InvalidOperationException
                        || exception is System.IO.IOException)
                    {
                        CloseConnection(ref stream, ref client);
                        RestorePayloadUnlessSuperseded(payload);
                        Thread.Sleep(100);
                        frameReady.Set();
                    }
                }

                CloseConnection(ref stream, ref client);
            }

            /// <summary>
            /// Opens a low-latency TCP connection with a bounded connect wait.
            /// </summary>
            /// <returns>Connected TCP client with Nagle buffering disabled.</returns>
            private TcpClient Connect()
            {
                var client = new TcpClient { NoDelay = true, SendTimeout = 250 };
                IAsyncResult result = client.BeginConnect(host, port, null, null);
                bool connected = result.AsyncWaitHandle.WaitOne(500);
                result.AsyncWaitHandle.Close();
                if (!connected)
                {
                    client.Close();
                    throw new SocketException((int)SocketError.TimedOut);
                }

                client.EndConnect(result);
                return client;
            }

            /// <summary>
            /// Removes and returns the newest pending JSON payload.
            /// </summary>
            /// <returns>Latest payload, or null when no frame is pending.</returns>
            private string TakeLatestPayload()
            {
                lock (payloadLock)
                {
                    string payload = latestPayload;
                    latestPayload = null;
                    return payload;
                }
            }

            /// <summary>
            /// Requeues a failed frame only when a newer frame has not already replaced it.
            /// </summary>
            /// <param name="payload">JSON frame that failed to send.</param>
            /// <returns>No value.</returns>
            private void RestorePayloadUnlessSuperseded(string payload)
            {
                lock (payloadLock)
                {
                    if (latestPayload == null)
                    {
                        latestPayload = payload;
                    }
                }
            }

            /// <summary>
            /// Closes network resources and clears their references.
            /// </summary>
            /// <param name="stream">Current TCP stream, possibly null.</param>
            /// <param name="client">Current TCP client, possibly null.</param>
            /// <returns>No value.</returns>
            private static void CloseConnection(ref NetworkStream stream, ref TcpClient client)
            {
                stream?.Dispose();
                client?.Close();
                stream = null;
                client = null;
            }

            /// <summary>
            /// Stops the worker and releases its wait handle.
            /// </summary>
            /// <returns>No value.</returns>
            public void Dispose()
            {
                running = false;
                frameReady.Set();
                worker?.Join(1000);
                frameReady.Dispose();
            }
        }
    }
}
