const state = {
  snapshot: null,
  viewerUrl: null,
  socket: null,
  settingsInitialized: false,
  settingsDirty: false,
  settingsApplying: false,
  settingsFeedback: null,
};

const actionButtons = Object.fromEntries(
  [...document.querySelectorAll("[data-action]")].map((button) => [button.dataset.action, button]),
);
const rlActionButtons = Object.fromEntries(
  [...document.querySelectorAll("[data-rl-action]")].map((button) => [button.dataset.rlAction, button]),
);
const evaluationActionButtons = Object.fromEntries(
  [...document.querySelectorAll("[data-evaluation-action]")].map((button) => [
    button.dataset.evaluationAction,
    button,
  ]),
);

function statusClass(value) {
  return `status-${String(value || "unknown").toLowerCase()}`;
}

function setStatusPill(element, value) {
  element.className = `status-pill ${statusClass(value)}`;
  element.textContent = value || "UNKNOWN";
}

function render(snapshot) {
  state.snapshot = snapshot;
  renderMode(snapshot);
  setStatusPill(document.querySelector("#system-status"), snapshot.system.state);
  setStatusPill(document.querySelector("#recording-state"), snapshot.recording.state);
  document.querySelector("#episode-index").textContent = snapshot.recording.current_episode ?? 0;
  document.querySelector("#episodes-complete").textContent = `${snapshot.recording.completed_episodes ?? 0} / ${snapshot.recording.target_episodes ?? 0}`;
  document.querySelector("#frame-count").textContent = snapshot.recording.frame_count ?? 0;
  document.querySelector("#elapsed-time").textContent = `${Number(snapshot.recording.elapsed_sec ?? 0).toFixed(1)} s`;
  document.querySelector("#reset-state").textContent = snapshot.reset.state || "UNKNOWN";
  document.querySelector("#session-state").textContent = snapshot.session.state || "UNKNOWN";
  document.querySelector("#recording-operation").textContent =
    snapshot.recording.operation_detail || "等待录制组件";
  document.querySelector("#instruction").textContent = snapshot.recording.instruction || "-";
  document.querySelector("#dataset-root").textContent = snapshot.recording.dataset_root || "等待数据集信息";
  renderSessionSettings(snapshot.recording);

  actionButtons.staging.disabled = !snapshot.recording.can_stage;
  actionButtons.start.disabled = !snapshot.recording.can_start;
  actionButtons.save.disabled = !snapshot.recording.can_save;
  actionButtons.optimal.disabled = !snapshot.recording.can_save;
  actionButtons.failure.disabled = !snapshot.recording.can_save;
  actionButtons.discard.disabled = !snapshot.recording.can_discard;
  actionButtons.finish.disabled = !snapshot.recording.can_finish;
  actionButtons.discard.querySelector("span").textContent =
    snapshot.recording.state === "STAGING" ? "放弃预定位" : "丢弃本条";
  renderHardware(snapshot.hardware);
  renderErrors(snapshot.errors);
  updateViewer(snapshot.viewer);
}

function renderMode(snapshot) {
  const mode = snapshot.mode || { name: "teleop", safety_state: "UNKNOWN", inference_enabled: false };
  const titles = {
    teleop: "遥操作录制控制台",
    inference: "远程推理操作台",
    rl: "RL Rollout 与人类接管操作台",
  };
  document.body.classList.toggle("runtime-nonteleop", mode.name !== "teleop");
  document.querySelector("#page-title").textContent = titles[mode.name] || "机器人操作台";
  setStatusPill(document.querySelector("#mode-status"), String(mode.name).toUpperCase());
  setStatusPill(document.querySelector("#motion-state"), mode.safety_state);
  document.querySelector("#operation-mode").textContent = titles[mode.name] || mode.name;
  document.querySelector("#operation-detail").textContent = mode.motion_enabled
    ? "真机运动已由启动参数显式解锁；Web 页面不能修改该状态。"
    : "DRY RUN：执行器未解锁，Web 页面不能在线开启真机运动。";

  const teleop = mode.name === "teleop";
  const vr = mode.teleop_source === "vr";
  const vrDetail = document.querySelector("#vr-detail");
  vrDetail.hidden = !vr;
  const vrHardware = (snapshot.hardware || []).find((component) => component.id === "vr");
  vrDetail.textContent = vr
    ? `VR TCP ${mode.motion_enabled ? "真机" : "预览"}：${snapshot.recording.vr_reason || vrHardware?.message || "等待手柄跟踪与输入"}`
    : "";
  const evaluation = snapshot.evaluation || { enabled: false, state: "DISABLED" };
  document.querySelector("#teleop-settings").hidden = !(teleop || evaluation.enabled || mode.name === "rl");
  document.querySelector("#teleop-recording").hidden = !(teleop || evaluation.enabled);
  document.querySelector("#teleop-controls").hidden = !teleop;

  const inference = snapshot.inference || { enabled: false, phase: "DISABLED", can_stop: false };
  document.querySelector("#inference-panel").hidden = !inference.enabled;
  document.querySelector("#inference-phase").textContent = inference.phase || "UNKNOWN";
  document.querySelector("#inference-phase").title = inference.reason || "";
  document.querySelector("#inference-source").textContent = inference.enabled ? "ENABLED" : "DISABLED";
  document.querySelector("#inference-stop").hidden = evaluation.enabled || mode.name === "rl";
  document.querySelector("#inference-stop").disabled =
    evaluation.enabled || mode.name === "rl" || !inference.can_stop;

  document.querySelector("#evaluation-panel").hidden = !evaluation.enabled;
  document.querySelector("#evaluation-state").textContent = evaluation.state || "UNKNOWN";
  document.querySelector("#evaluation-remaining").textContent = `${Number(evaluation.remaining_sec ?? 0).toFixed(1)} s`;
  document.querySelector("#evaluation-successes").textContent = evaluation.successful_episodes ?? 0;
  document.querySelector("#evaluation-failures").textContent = evaluation.failed_episodes ?? 0;
  document.querySelector("#evaluation-discards").textContent = evaluation.discarded_attempts ?? 0;
  document.querySelector("#evaluation-last-outcome").textContent = evaluation.last_outcome
    ? `${evaluation.last_outcome}: ${evaluation.last_outcome_reason || "-"}`
    : "-";
  document.querySelector("#evaluation-start-blocker").textContent = evaluation.can_start
    ? "已就绪"
    : (evaluation.start_blocker || "等待评测组件");
  document.querySelector("#evaluation-operation").textContent =
    evaluation.operation_detail || "等待录制组件";
  evaluationActionButtons.start.disabled = !evaluation.can_start;
  evaluationActionButtons.success.disabled = !evaluation.can_label;
  evaluationActionButtons.failure.disabled = !evaluation.can_label;
  evaluationActionButtons.discard.disabled = !evaluation.can_discard;
  evaluationActionButtons.finish.disabled = !evaluation.can_finish;

  const rl = snapshot.rl || { enabled: false, state: "DISABLED" };
  document.querySelector("#rl-panel").hidden = !rl.enabled;
  document.querySelector("#rl-state").textContent = rl.state || "UNKNOWN";
  document.querySelector("#rl-source").textContent = rl.action_source || "unknown";
  document.querySelector("#rl-human-ready").textContent =
    rl.human_input_ready ? "已就绪" : "等待设备持续在线、输入就绪";
  document.querySelector("#rl-intervention").textContent = rl.intervention_active ? "是" : "否";
  document.querySelector("#rl-episode-state").textContent = rl.episode_state || "UNKNOWN";
  document.querySelector("#rl-capture-progress").textContent =
    `${rl.frame_count ?? 0} / ${Number(rl.elapsed_sec ?? 0).toFixed(1)} s`;
  document.querySelector("#rl-episode-counts").textContent =
    `${rl.success_episodes ?? 0} / ${rl.failure_episodes ?? 0} / ${rl.crash_episodes ?? 0}`;
  document.querySelector("#rl-last-outcome").textContent = rl.last_outcome
    ? `${rl.last_outcome}: ${rl.last_outcome_reason || "-"}` : "-";
  document.querySelector("#rl-operation").textContent =
    rl.operation_detail || rl.start_blocker || "等待录制组件";
  rlActionButtons.start.disabled = !rl.can_start;
  rlActionButtons.discard.disabled = !rl.can_discard;
  rlActionButtons.finish.disabled = !rl.can_finish;
  const reward = rl.last_reward;
  document.querySelector("#rl-last-reward").textContent = reward
    ? `${reward.event || "unknown"} (${reward.reward ?? "-"})`
    : "-";

  const switchLabels = {
    POLICY: "切换到人工",
    PAUSED: "切换到人工",
    STOPPING: "正在切换到人工…",
    HUMAN_CONTROL: "切换到策略",
  };
  rlActionButtons.switch.querySelector("span").textContent = switchLabels[rl.state] || "切换控制权";
  rlActionButtons.switch.disabled = !rl.can_switch;
  for (const action of ["success", "failure"]) {
    rlActionButtons[action].disabled = !rl.can_reward;
  }
  rlActionButtons.stop.disabled = !rl.can_stop;
}

function renderSessionSettings(recording) {
  const instruction = document.querySelector("#session-instruction");
  const datasetRoot = document.querySelector("#session-dataset-root");
  const modeInputs = [...document.querySelectorAll("input[name='write-mode']")];
  const configured = Boolean(recording.session_configured);
  if (!state.settingsInitialized || (configured && !state.settingsDirty)) {
    instruction.value = recording.instruction || "";
    datasetRoot.value = recording.dataset_root || "";
    for (const input of modeInputs) input.checked = input.value === (recording.write_mode || "resume");
    state.settingsInitialized = true;
  }
  const editable = Boolean(recording.can_configure);
  instruction.disabled = !editable;
  datasetRoot.disabled = !editable;
  for (const input of modeInputs) input.disabled = !editable;
  document.querySelector("#apply-session-settings").disabled = !editable || state.settingsApplying;
  setStatusPill(
    document.querySelector("#session-config-status"),
    configured ? "LOCKED" : "EDITABLE",
  );
  const feedback = document.querySelector("#session-feedback");
  if (!configured && recording.last_error) {
    setSessionFeedback(`上次配置失败：${recording.last_error}`, "error");
  } else if (configured) {
    const mode = recording.write_mode === "overwrite" ? "Overwrite" : "Resume";
    setSessionFeedback(`已确认 → ${recording.dataset_root}（${mode}）`, "ok");
  } else if (state.settingsFeedback) {
    setSessionFeedback(state.settingsFeedback.message, state.settingsFeedback.kind);
  } else {
    feedback.hidden = true;
  }
}

function renderHardware(hardware) {
  const target = document.querySelector("#hardware-list");
  target.replaceChildren();
  if (!hardware.length) {
    target.append(emptyState("等待硬件诊断"));
    return;
  }
  for (const component of hardware) {
    const row = document.createElement("div");
    row.className = "hardware-row";
    const dot = document.createElement("span");
    dot.className = `health-dot ${String(component.state || "unknown").toLowerCase()}`;
    const name = document.createElement("span");
    name.className = "hardware-name";
    name.textContent = component.id;
    const message = document.createElement("span");
    message.className = "hardware-message";
    message.textContent = component.state === "DISABLED" ? "YAML 已禁用" : component.message;
    row.append(dot, name, message);
    target.append(row);
  }
}

function renderErrors(errors) {
  const target = document.querySelector("#error-list");
  document.querySelector("#error-count").textContent = errors.length;
  target.replaceChildren();
  if (!errors.length) {
    target.append(emptyState("暂无警告或错误"));
    return;
  }
  for (const event of [...errors].reverse().slice(0, 30)) {
    const item = document.createElement("div");
    item.className = "error-item";
    const heading = document.createElement("strong");
    heading.textContent = `${event.node || "unknown"} · ${new Date(event.timestamp * 1000).toLocaleTimeString()}`;
    const message = document.createElement("p");
    message.textContent = event.message;
    item.append(heading, message);
    target.append(item);
  }
}

function emptyState(text) {
  const element = document.createElement("div");
  element.className = "empty-state";
  element.textContent = text;
  return element;
}

function updateViewer(viewer) {
  const iframe = document.querySelector("#rerun-viewer");
  const placeholder = document.querySelector("#viewer-placeholder");
  if (!viewer.enabled || !viewer.ready) {
    iframe.style.display = "none";
    placeholder.style.display = "grid";
    placeholder.textContent = viewer.enabled ? "等待 Rerun 服务启动…" : "Rerun 已在 YAML 中禁用";
    return;
  }
  if (state.viewerUrl !== viewer.url) {
    iframe.src = viewer.url;
    state.viewerUrl = viewer.url;
  }
  iframe.style.display = "block";
  placeholder.style.display = "none";
}

async function performAction(action) {
  const button = actionButtons[action];
  if (!button || button.disabled) return;
  button.disabled = true;
  try {
    const response = await fetch(`/api/v1/recording/${action}`, { method: "POST" });
    const result = await response.json();
    showToast(result.message, !result.ok);
  } catch (error) {
    showToast(`控制请求失败: ${error}`, true);
  }
}

async function performInferenceStop() {
  if (document.querySelector("#inference-stop").disabled) return;
  if (!window.confirm("停止当前推理 rollout？停止后必须重新启动 launch 才能恢复。")) return;
  await postOperatorAction("/api/v1/inference/stop", "停止推理");
}

async function performEvaluationAction(action) {
  const button = evaluationActionButtons[action];
  if (!button || button.disabled) return;
  const confirmations = {
    success: "确认当前 rollout 成功，并保存本条数据？",
    failure: "确认当前 rollout 失败，并保存本条数据？",
    discard: "确认丢弃当前 rollout？本条数据不会保存。",
    finish: "确认结束评测会话？",
  };
  if (confirmations[action] && !window.confirm(confirmations[action])) return;
  const paths = {
    start: "/api/v1/recording/start",
    success: "/api/v1/evaluation/success",
    failure: "/api/v1/evaluation/failure",
    discard: "/api/v1/recording/discard",
    finish: "/api/v1/recording/finish",
  };
  await postOperatorAction(paths[action], `评测 ${action}`);
}

async function performRlAction(action) {
  const button = rlActionButtons[action];
  if (!button || button.disabled) return;
  const confirmations = {
    success: "确认将当前 rollout 标记为成功并结束本条？",
    failure: "确认将当前 rollout 标记为失败并结束本条？",
    discard: "确认丢弃当前 RL episode？本条数据不会保存。",
    finish: "确认结束 RL 会话？当前未结束的轨迹会按失败保存并复位。",
  };
  if (confirmations[action] && !window.confirm(confirmations[action])) return;
  await postOperatorAction(`/api/v1/rl/${action}`, `RL ${action}`);
}

async function postOperatorAction(path, label) {
  showToast(`${label}请求已发送，等待系统处理…`, false);
  try {
    const response = await fetch(path, { method: "POST" });
    const result = await response.json();
    showToast(result.message || label, !result.ok);
    return Boolean(result.ok);
  } catch (error) {
    showToast(`${label}失败: ${error}`, true);
    return false;
  }
}

async function applySessionSettings() {
  if (state.settingsApplying) return false;
  const instruction = document.querySelector("#session-instruction").value.trim();
  const datasetRoot = document.querySelector("#session-dataset-root").value.trim();
  const writeMode = document.querySelector("input[name='write-mode']:checked")?.value || "resume";
  if (!instruction || !datasetRoot) {
    setSessionFeedback("任务指令和数据保存目录不能为空", "error");
    showToast("任务指令和数据集路径不能为空", true);
    return false;
  }
  if (writeMode === "overwrite" && !window.confirm(`确认清空并覆盖该目录中的录制分类？\n${datasetRoot}`)) {
    setSessionFeedback("已取消 Overwrite，录制设置未修改", "info");
    return false;
  }
  const applyButton = document.querySelector("#apply-session-settings");
  state.settingsApplying = true;
  applyButton.disabled = true;
  applyButton.textContent = "应用中…";
  setSessionFeedback("正在验证并应用录制设置…", "info");
  try {
    const response = await fetch("/api/v1/recording/configure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction, dataset_root: datasetRoot, write_mode: writeMode }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
    setSessionFeedback(result.message, result.ok ? "ok" : "error");
    showToast(result.message, !result.ok);
    if (result.ok) state.settingsDirty = false;
    return Boolean(result.ok);
  } catch (error) {
    setSessionFeedback(`录制设置失败: ${error}`, "error");
    showToast(`录制设置失败: ${error}`, true);
    return false;
  } finally {
    state.settingsApplying = false;
    applyButton.textContent = "应用设置";
    const recording = state.snapshot?.recording;
    applyButton.disabled = !recording?.can_configure || Boolean(recording?.session_configured);
  }
}

function setSessionFeedback(message, kind) {
  state.settingsFeedback = { message, kind };
  const feedback = document.querySelector("#session-feedback");
  feedback.hidden = false;
  feedback.textContent = message;
  feedback.className = `session-feedback ${kind}`;
}

function showToast(message, isError = false) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.className = `toast visible${isError ? " error" : ""}`;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { toast.className = "toast"; }, 2800);
}

function connectEvents() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/api/v1/events`);
  state.socket = socket;
  socket.addEventListener("open", () => {
    document.querySelector("#connection-dot").classList.add("connected");
    document.querySelector("#connection-label").textContent = "实时连接";
  });
  socket.addEventListener("message", (event) => render(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    document.querySelector("#connection-dot").classList.remove("connected");
    document.querySelector("#connection-label").textContent = "连接中断，正在重试";
    window.setTimeout(connectEvents, 1200);
  });
}

for (const button of Object.values(actionButtons)) {
  button.addEventListener("click", () => performAction(button.dataset.action));
}
for (const button of Object.values(rlActionButtons)) {
  button.addEventListener("click", () => performRlAction(button.dataset.rlAction));
}
for (const button of Object.values(evaluationActionButtons)) {
  button.addEventListener("click", () => performEvaluationAction(button.dataset.evaluationAction));
}
document.querySelector("#inference-stop").addEventListener("click", performInferenceStop);

for (const input of document.querySelectorAll("#session-instruction, #session-dataset-root, input[name='write-mode']")) {
  input.addEventListener("input", () => { state.settingsDirty = true; });
  input.addEventListener("change", () => { state.settingsDirty = true; });
}

document.querySelector("#apply-session-settings").addEventListener("click", applySessionSettings);

document.addEventListener("keydown", (event) => {
  if (event.repeat || event.target.matches("input, textarea, select, [contenteditable='true']")) return;
  const mode = state.snapshot?.mode?.name;
  if (mode === "teleop") {
    const action = {
      g: "staging",
      G: "staging",
      s: "start",
      S: "start",
      ArrowLeft: "discard",
      ArrowRight: "save",
      o: "optimal",
      O: "optimal",
      f: "failure",
      F: "failure",
      Escape: "finish",
    }[event.key];
    if (action) {
      event.preventDefault();
      performAction(action);
    }
    return;
  }
  if (mode === "inference") {
    const evaluation = state.snapshot?.evaluation;
    if (evaluation?.enabled) {
      const action = {
        s: "start",
        S: "start",
        y: "success",
        Y: "success",
        n: "failure",
        N: "failure",
        ArrowLeft: "discard",
        Escape: "finish",
      }[event.key];
      if (action) {
        event.preventDefault();
        performEvaluationAction(action);
      }
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      performInferenceStop();
    }
    return;
  }
  if (mode === "rl") {
    const action = {
      s: "start",
      S: "start",
      q: "finish",
      Q: "finish",
      ArrowLeft: "discard",
      t: "switch",
      T: "switch",
      y: "success",
      Y: "success",
      n: "failure",
      N: "failure",
      Escape: "stop",
    }[event.key];
    if (action) {
      event.preventDefault();
      performRlAction(action);
    }
  }
});

document.querySelector("#reload-viewer").addEventListener("click", () => {
  const iframe = document.querySelector("#rerun-viewer");
  if (iframe.src) iframe.src = iframe.src;
});

fetch("/api/v1/status").then((response) => response.json()).then(render);
connectEvents();
