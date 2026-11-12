using System;
using System.IO;
using System.Linq;
using ControlRobot.QuestPose;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Creates and builds the minimal Quest scene used to validate controller poses.
/// </summary>
public static class QuestPoseBridgeSetup
{
    private const string SceneDirectory = "Assets/Scenes";
    private const string ScenePath = SceneDirectory + "/Main.unity";
    private const string ApkPath = "Builds/QuestPoseBridge.apk";

    /// <summary>
    /// Creates a scene containing OVRCameraRig, pose streaming, lighting, and spatial references.
    /// </summary>
    /// <returns>No value. The scene is saved at Assets/Scenes/Main.unity.</returns>
    public static void CreateScene()
    {
        EnsureAssetDirectory(SceneDirectory);
        Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        GameObject rigObject = InstantiateCameraRig();
        OVRCameraRig cameraRig = rigObject.GetComponent<OVRCameraRig>();
        cameraRig.EnsureGameObjectIntegrity();

        // Record a prefab override so floor-level tracking survives scene serialization.
        var managerSettings = new SerializedObject(rigObject.GetComponent<OVRManager>());
        managerSettings.FindProperty("_trackingOriginType").intValue = (int)OVRManager.TrackingOrigin.FloorLevel;
        managerSettings.ApplyModifiedProperties();

        GameObject streamerObject = new GameObject("ControllerPoseStreamer");
        ControllerPoseLogger streamer = streamerObject.AddComponent<ControllerPoseLogger>();
        streamer.Configure(
            cameraRig.trackingSpace,
            cameraRig.leftHandAnchor,
            cameraRig.rightHandAnchor);

        CreateSpatialReferences();
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene, ScenePath);

        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        PlayerSettings.productName = "Quest Pose Bridge";
        PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, "com.controlrobot.questposebridge");
        PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
        PlayerSettings.Android.forceInternetPermission = true;
        OVRManifestPreprocessor.GenerateOrUpdateAndroidManifest(true);
        AssetDatabase.SaveAssets();

        Debug.Log($"QUEST_POSE_SETUP scene={ScenePath} frame={cameraRig.trackingSpace.name} origin=FloorLevel "
            + $"left={cameraRig.leftHandAnchor.name} right={cameraRig.rightHandAnchor.name}");
    }

    /// <summary>
    /// Creates the validation scene and builds an Android development APK.
    /// </summary>
    /// <returns>No value. A failed Unity build throws an exception.</returns>
    public static void BuildAndroid()
    {
        CreateScene();
        Directory.CreateDirectory(Path.GetDirectoryName(ApkPath) ?? "Builds");

        var options = new BuildPlayerOptions
        {
            scenes = new[] { ScenePath },
            locationPathName = ApkPath,
            target = BuildTarget.Android,
            options = BuildOptions.Development
        };

        BuildReport report = BuildPipeline.BuildPlayer(options);
        if (report.summary.result != BuildResult.Succeeded)
        {
            throw new InvalidOperationException(
                $"Quest APK build failed: {report.summary.result}, errors={report.summary.totalErrors}");
        }

        Debug.Log($"QUEST_POSE_BUILD apk={Path.GetFullPath(ApkPath)} bytes={report.summary.totalSize}");
    }

    /// <summary>
    /// Loads the Meta XR Core SDK OVRCameraRig prefab into the active scene.
    /// </summary>
    /// <returns>The instantiated rig root GameObject.</returns>
    private static GameObject InstantiateCameraRig()
    {
        string prefabPath = AssetDatabase.FindAssets("OVRCameraRig t:Prefab")
            .Select(AssetDatabase.GUIDToAssetPath)
            .First(path => path.Contains("com.meta.xr.sdk.core") && path.EndsWith("OVRCameraRig.prefab"));

        GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
        GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        instance.name = "OVRCameraRig";
        instance.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
        return instance;
    }

    /// <summary>
    /// Adds a floor, a forward cube, and a directional light for visual orientation in the headset.
    /// </summary>
    /// <returns>No value.</returns>
    private static void CreateSpatialReferences()
    {
        GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.localScale = new Vector3(2f, 1f, 2f);

        GameObject referenceCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        referenceCube.name = "ForwardReference";
        referenceCube.transform.position = new Vector3(0f, 1.2f, 2f);
        referenceCube.transform.localScale = Vector3.one * 0.25f;

        GameObject lightObject = new GameObject("Directional Light");
        Light light = lightObject.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.2f;
        lightObject.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
    }

    /// <summary>
    /// Creates an Assets-relative directory hierarchy when it does not already exist.
    /// </summary>
    /// <param name="assetDirectory">Directory such as Assets/Scenes.</param>
    /// <returns>No value.</returns>
    private static void EnsureAssetDirectory(string assetDirectory)
    {
        string[] parts = assetDirectory.Split('/');
        string parent = parts[0];
        for (int index = 1; index < parts.Length; index++)
        {
            string childPath = parent + "/" + parts[index];
            if (!AssetDatabase.IsValidFolder(childPath))
            {
                AssetDatabase.CreateFolder(parent, parts[index]);
            }

            parent = childPath;
        }
    }
}
