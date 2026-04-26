const appState = {
  map: null,
  overlay: null,
  status: null,
  draftRoute: [],
  patrolRoute: [],
  patrolState: null,
  routeMode: false,
  view: {
    scale: 1,
    offsetX: 0,
    offsetY: 0,
  },
  dragging: false,
  dragStart: null,
};

const canvas = document.getElementById("mapCanvas");
const ctx = canvas.getContext("2d");

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.floor(rect.width * window.devicePixelRatio);
  canvas.height = Math.floor(rect.height * window.devicePixelRatio);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  draw();
}

function updateStatusPanel(status) {
  document.getElementById("navLog").textContent = status.last_nav_log || status.nav_detail || "暂无导航日志";

  if (status.pose) {
    const poseText = `x=${status.pose.x.toFixed(2)} y=${status.pose.y.toFixed(2)} yaw=${status.pose.yaw.toFixed(2)}`;
    document.getElementById("floatingPose").textContent = poseText;
  } else {
    document.getElementById("floatingPose").textContent = "-";
  }

  if (status.speed) {
    const speedText = `v=${status.speed.linear.toFixed(2)} w=${status.speed.angular.toFixed(2)}`;
    document.getElementById("floatingSpeed").textContent = speedText;
  } else {
    document.getElementById("floatingSpeed").textContent = "-";
  }

  document.getElementById("floatingNavStatus").textContent = status.nav_status || "-";
  document.getElementById("floatingTrajectoryCount").textContent = String(status.trajectory_count || 0);
  document.getElementById("floatingRouteSummary").textContent =
    `${status.route_waypoint_count || 0} / 当前点 ${(status.current_waypoint_index ?? -1) + 1}`;

  document.getElementById("mapStatus").textContent = status.map_available
    ? `地图已加载 #${status.map_seq}`
    : "地图未加载";
  document.getElementById("cameraStatus").textContent = status.camera_available
    ? `相机在线 ${new Date(status.camera_stamp * 1000).toLocaleTimeString()}`
    : "相机未连接";

  const detectionBtn = document.getElementById("toggleDetection");
  const detEnabled = status.detection_enabled;
  detectionBtn.textContent = detEnabled ? "\u{1F3AF} 隐藏检测" : "\u{1F3AF} 显示检测";
  detectionBtn.classList.toggle("active", detEnabled);
}

function updateRouteList() {
  const list = document.getElementById("routeList");
  list.innerHTML = "";
  appState.draftRoute.forEach((point, index) => {
    const li = document.createElement("li");
    li.textContent = `${index + 1}. (${point.x.toFixed(2)}, ${point.y.toFixed(2)})`;
    list.appendChild(li);
  });

  document.getElementById("toggleRouteMode").textContent = appState.routeMode ? "结束选点" : "开始选点";
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  return response.json();
}

async function refreshStatus() {
  const data = await fetchJSON("/api/status");
  appState.status = data;
  updateStatusPanel(data);
}

async function refreshMap() {
  try {
    const data = await fetchJSON("/api/map");
    if (data.ok && (!appState.map || appState.map.seq !== data.seq)) {
      appState.map = data;
      fitMapToView();
      draw();
    }
  } catch (_err) {
  }
}

async function refreshOverlay() {
  const data = await fetchJSON("/api/overlay");
  appState.overlay = data;
  draw();
}

function refreshCameraFrame() {
  const img = document.getElementById("cameraFeed");
  if (!img.dataset.streaming) {
    img.src = "/api/camera/stream";
    img.dataset.streaming = "1";
  }
}

async function refreshDetectionStats() {
  try {
    const data = await fetchJSON("/api/detection");
    if (!data.ok) return;

    document.getElementById("detectionCount").textContent = `${data.count} 个目标`;
    document.getElementById("detectionCount").style.display = data.enabled ? "" : "none";

    const list = document.getElementById("detectionList");
    if (data.enabled && data.count > 0) {
      list.innerHTML = data.objects.map((obj, i) => {
        const pct = (obj.confidence * 100).toFixed(0);
        return `<div class="detection-item"><span class="detection-label">${i + 1}. ${obj.label}</span><span class="detection-conf">${pct}%</span></div>`;
      }).join("");
    } else if (data.enabled) {
      list.innerHTML = '<span class="hint">未检测到目标</span>';
    } else {
      list.innerHTML = '<span class="hint">检测已关闭，点击上方按钮开启</span>';
    }
  } catch (_err) {
    // ignore
  }
}

function fitMapToView() {
  if (!appState.map) {
    return;
  }

  const rect = canvas.getBoundingClientRect();
  const mapPixelWidth = appState.map.width;
  const mapPixelHeight = appState.map.height;
  const scaleX = rect.width / mapPixelWidth;
  const scaleY = rect.height / mapPixelHeight;
  appState.view.scale = Math.min(scaleX, scaleY) * 0.92;
  appState.view.offsetX = (rect.width - mapPixelWidth * appState.view.scale) / 2;
  appState.view.offsetY = (rect.height - mapPixelHeight * appState.view.scale) / 2;
}

function worldToScreen(x, y) {
  const map = appState.map;
  if (!map) {
    return {x: 0, y: 0};
  }

  const px = (x - map.origin.x) / map.resolution;
  const py = (y - map.origin.y) / map.resolution;
  const sx = appState.view.offsetX + px * appState.view.scale;
  const sy = appState.view.offsetY + (map.height - py) * appState.view.scale;
  return {x: sx, y: sy};
}

function screenToWorld(x, y) {
  const map = appState.map;
  const px = (x - appState.view.offsetX) / appState.view.scale;
  const py = map.height - (y - appState.view.offsetY) / appState.view.scale;
  return {
    x: map.origin.x + px * map.resolution,
    y: map.origin.y + py * map.resolution,
  };
}

function drawMap() {
  if (!appState.map) {
    ctx.fillStyle = "#5e7067";
    ctx.font = "20px sans-serif";
    ctx.fillText("等待 /map 数据...", 30, 40);
    return;
  }

  const map = appState.map;
  const imageData = ctx.createImageData(map.width, map.height);
  for (let y = 0; y < map.height; y += 1) {
    for (let x = 0; x < map.width; x += 1) {
      const srcIndex = x + (map.height - 1 - y) * map.width;
      const dstIndex = (x + y * map.width) * 4;
      const value = map.data[srcIndex];

      let color = 205;
      if (value === -1) {
        color = 235;
      } else if (value >= 65) {
        color = 45;
      } else if (value >= 0) {
        color = 255 - Math.round(value * 1.8);
      }

      imageData.data[dstIndex] = color;
      imageData.data[dstIndex + 1] = color;
      imageData.data[dstIndex + 2] = color;
      imageData.data[dstIndex + 3] = 255;
    }
  }

  const buffer = document.createElement("canvas");
  buffer.width = map.width;
  buffer.height = map.height;
  buffer.getContext("2d").putImageData(imageData, 0, 0);
  ctx.drawImage(
    buffer,
    appState.view.offsetX,
    appState.view.offsetY,
    map.width * appState.view.scale,
    map.height * appState.view.scale
  );
}

function drawPolyline(points, color, width) {
  if (!points || points.length < 2) {
    return;
  }

  ctx.beginPath();
  const start = worldToScreen(points[0].x, points[0].y);
  ctx.moveTo(start.x, start.y);
  for (let i = 1; i < points.length; i += 1) {
    const p = worldToScreen(points[i].x, points[i].y);
    ctx.lineTo(p.x, p.y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();
}

function drawWaypoints(points, activeIndex = -1, color = "#355f8d") {
  if (!points) {
    return;
  }

  points.forEach((point, index) => {
    const p = worldToScreen(point.x, point.y);
    ctx.beginPath();
    ctx.arc(p.x, p.y, index === activeIndex ? 8 : 6, 0, Math.PI * 2);
    ctx.fillStyle = index === activeIndex ? "#d1495b" : color;
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "12px sans-serif";
    ctx.fillText(String(index + 1), p.x - 3, p.y + 4);
  });
}

function drawRobot(pose) {
  if (!pose) {
    return;
  }

  const p = worldToScreen(pose.x, pose.y);
  ctx.save();
  ctx.translate(p.x, p.y);
  ctx.rotate(-pose.yaw);
  ctx.fillStyle = "#d1495b";
  ctx.beginPath();
  ctx.moveTo(14, 0);
  ctx.lineTo(-10, -8);
  ctx.lineTo(-6, 0);
  ctx.lineTo(-10, 8);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawGrid() {
  const rect = canvas.getBoundingClientRect();
  ctx.strokeStyle = "rgba(60, 90, 78, 0.08)";
  ctx.lineWidth = 1;
  for (let x = 0; x < rect.width; x += 50) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, rect.height);
    ctx.stroke();
  }
  for (let y = 0; y < rect.height; y += 50) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(rect.width, y);
    ctx.stroke();
  }
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  drawGrid();
  drawMap();

  if (appState.overlay) {
    drawPolyline(appState.overlay.plan, "#355f8d", 3);
    drawPolyline(appState.overlay.trajectory, "#db8f2f", 2);
    drawPolyline(appState.overlay.current_route, "#5b4ab8", 2);
    drawWaypoints(appState.overlay.current_route, appState.overlay.current_waypoint_index, "#5b4ab8");
    drawRobot(appState.overlay.pose);
  }

  drawPolyline(appState.draftRoute, "#1d5c4e", 2);
  drawWaypoints(appState.draftRoute, -1, "#1d5c4e");

  drawPatrolDetections();
}

document.getElementById("commandForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = document.getElementById("commandInput").value.trim();
  if (!text) {
    return;
  }

  const result = await fetchJSON("/api/command", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text}),
  });
  document.getElementById("commandResult").textContent = result.message || result.error || "已发送";
});

document.getElementById("toggleRouteMode").addEventListener("click", () => {
  appState.routeMode = !appState.routeMode;
  updateRouteList();
});

document.getElementById("toggleCameraPanel").addEventListener("click", () => {
  const shell = document.getElementById("cameraShell");
  shell.classList.toggle("collapsed");
  document.getElementById("toggleCameraPanel").textContent =
    shell.classList.contains("collapsed") ? "展开" : "折叠";
});

document.getElementById("toggleDetection").addEventListener("click", async () => {
  const status = appState.status;
  const enabled = status ? !status.detection_enabled : true;
  await fetchJSON("/api/detection/toggle", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({enable: enabled}),
  });
  // Refresh status to update button text immediately
  await refreshStatus();
});

document.getElementById("clearDraft").addEventListener("click", () => {
  appState.draftRoute = [];
  updateRouteList();
  draw();
});

document.getElementById("clearTrajectory").addEventListener("click", async () => {
  await fetchJSON("/api/trajectory/clear", {method: "POST"});
});

document.getElementById("cancelRoute").addEventListener("click", async () => {
  await fetchJSON("/api/route/cancel", {method: "POST"});
});

document.getElementById("sendRoute").addEventListener("click", async () => {
  if (!appState.draftRoute.length) {
    document.getElementById("commandResult").textContent = "请先在地图上添加至少一个 waypoint";
    return;
  }

  try {
    const result = await fetchJSON("/api/route", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({waypoints: appState.draftRoute}),
    });

    document.getElementById("commandResult").textContent =
      result.message || result.error || "路线已下发";
  } catch (_err) {
    document.getElementById("commandResult").textContent = "路线下发失败，请检查 dashboard 后端日志";
  }
});

canvas.addEventListener("click", (event) => {
  if (!appState.routeMode || !appState.map) {
    return;
  }

  const rect = canvas.getBoundingClientRect();
  const world = screenToWorld(event.clientX - rect.left, event.clientY - rect.top);
  appState.draftRoute.push(world);
  updateRouteList();
  draw();
});

canvas.addEventListener("mousedown", (event) => {
  appState.dragging = true;
  appState.dragStart = {
    x: event.clientX,
    y: event.clientY,
    offsetX: appState.view.offsetX,
    offsetY: appState.view.offsetY,
  };
});

window.addEventListener("mouseup", () => {
  appState.dragging = false;
});

window.addEventListener("mousemove", (event) => {
  if (!appState.dragging || !appState.dragStart) {
    return;
  }

  appState.view.offsetX = appState.dragStart.offsetX + (event.clientX - appState.dragStart.x);
  appState.view.offsetY = appState.dragStart.offsetY + (event.clientY - appState.dragStart.y);
  draw();
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const delta = event.deltaY < 0 ? 1.08 : 0.92;
  const rect = canvas.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const mouseY = event.clientY - rect.top;

  const oldScale = appState.view.scale;
  const newScale = Math.max(0.2, Math.min(oldScale * delta, 8));
  appState.view.scale = newScale;
  appState.view.offsetX = mouseX - ((mouseX - appState.view.offsetX) / oldScale) * newScale;
  appState.view.offsetY = mouseY - ((mouseY - appState.view.offsetY) / oldScale) * newScale;
  draw();
}, {passive: false});

window.addEventListener("resize", resizeCanvas);

// ---- patrol ----

async function refreshPatrolState() {
  try {
    const data = await fetchJSON("/api/patrol/state");
    if (!data.ok) return;
    appState.patrolState = data;
    updatePatrolPanel(data);
  } catch (_err) {
    // ignore
  }
}

function updatePatrolPanel(state) {
  const badge = document.getElementById("patrolStatusBadge");
  const progress = document.getElementById("patrolProgress");
  const info = document.getElementById("patrolInfo");

  badge.textContent = statusLabel(state.status);

  if (state.active) {
    const wp = state.current_index + 1;
    const total = state.waypoint_count;
    progress.textContent = `${wp} / ${total} 航点`;
    info.innerHTML = `<span class="patrol-person-count">检测到 <strong>${state.person_count}</strong> 人</span>`;
  } else if (state.status === "completed" || state.status === "partial") {
    progress.textContent = "已完成";
    info.innerHTML = `<span class="patrol-person-count">共检测到 <strong>${state.person_count}</strong> 人</span>`;
    if (state.report) {
      renderPatrolReport(state.report);
    }
  } else if (state.status === "aborted") {
    progress.textContent = "已中止";
    info.innerHTML = `<span class="patrol-person-count">检测到 <strong>${state.person_count}</strong> 人（部分报告）</span>`;
    if (state.report) {
      renderPatrolReport(state.report);
    }
  } else {
    progress.textContent = "";
    info.innerHTML = "";
  }

  document.getElementById("generatePatrolRoute").disabled = state.active;
  document.getElementById("startPatrol").disabled = state.active || appState.patrolRoute.length === 0;
  document.getElementById("stopPatrol").disabled = !state.active;
}

function statusLabel(s) {
  const labels = {
    idle: "就绪", generating: "生成路线中", route_ready: "路线已生成",
    running: "巡逻中", completed: "已完成", partial: "部分完成",
    aborted: "已中止",
  };
  return labels[s] || s;
}

async function handleGenerateRoute() {
  // Immediately show generating state
  document.getElementById("patrolStatusBadge").textContent = "正在生成路径";
  document.getElementById("generatePatrolRoute").disabled = true;

  const res = await fetchJSON("/api/patrol/generate", { method: "POST" });

  document.getElementById("generatePatrolRoute").disabled = false;

  if (!res.ok) {
    document.getElementById("commandResult").textContent = `路线生成失败: ${res.error}`;
    return;
  }
  appState.patrolRoute = res.waypoints || [];
  appState.draftRoute = res.waypoints || [];
  updateRouteList();
  document.getElementById("startPatrol").disabled = false;
  document.getElementById("commandResult").textContent =
    `自动路线已生成: ${res.waypoints.length} 个航点`;
  draw();
}

async function handleStartPatrol() {
  const res = await fetchJSON("/api/patrol/start", { method: "POST" });
  if (!res.ok) {
    document.getElementById("commandResult").textContent = `巡逻启动失败: ${res.error}`;
    return;
  }
  document.getElementById("commandResult").textContent = `巡逻已启动: ${res.message}`;
  // clear previous report
  document.getElementById("patrolReport").style.display = "none";
}

async function handleStopPatrol() {
  const res = await fetchJSON("/api/patrol/stop", { method: "POST" });
  document.getElementById("commandResult").textContent = res.message || "巡逻已停止";
}

function renderPatrolReport(report) {
  const panel = document.getElementById("patrolReport");
  const body = document.getElementById("patrolReportBody");
  panel.style.display = "block";

  const route = report.route_summary || {};
  const summary = report.summary || {};
  const persons = report.persons || [];
  const warnings = report.warnings || [];

  let html = `
    <div class="report-summary-row">
      <div class="report-stat">
        <span class="report-stat-value">${report.duration_seconds.toFixed(0)}s</span>
        <span class="report-stat-label">耗时</span>
      </div>
      <div class="report-stat">
        <span class="report-stat-value">${route.completed_waypoints}/${route.total_waypoints}</span>
        <span class="report-stat-label">航点</span>
      </div>
      <div class="report-stat">
        <span class="report-stat-value">${summary.total_person_detections}</span>
        <span class="report-stat-label">人员检测</span>
      </div>
      <div class="report-stat">
        <span class="report-stat-value">${summary.unique_locations}</span>
        <span class="report-stat-label">不同位置</span>
      </div>
    </div>
    <div class="report-meta">
      ${report.start_time} &ndash; ${report.end_time}<br>
      状态: ${statusLabel(report.status)}
    </div>`;

  if (warnings.length > 0) {
    html += `<div class="report-warnings"><strong>警告:</strong><ul>`;
    warnings.forEach(w => { html += `<li>${w}</li>`; });
    html += `</ul></div>`;
  }

  if (persons.length > 0) {
    html += `<div class="report-persons"><strong>人员检测记录:</strong><table class="report-table">
      <tr><th>#</th><th>时间</th><th>航点</th><th>位置</th><th>置信度</th><th>照片</th></tr>`;
    persons.forEach((p, i) => {
      const pos = (p.map_x != null && p.map_y != null)
        ? `(${p.map_x.toFixed(2)}, ${p.map_y.toFixed(2)})` : "—";
      const photo = p.photo
        ? `<a href="${p.photo}" target="_blank"><img src="${p.photo}" class="report-thumb"></a>`
        : "—";
      html += `<tr>
        <td>${i + 1}</td>
        <td>${p.time || "—"}</td>
        <td>${p.waypoint_index >= 0 ? p.waypoint_index + 1 : "—"}</td>
        <td>${pos}</td>
        <td>${(p.confidence * 100).toFixed(0)}%</td>
        <td>${photo}</td>
      </tr>`;
    });
    html += `</table></div>`;
  } else {
    html += `<div class="report-no-persons">未检测到人员</div>`;
  }

  body.innerHTML = html;
}

function drawPatrolDetections() {
  const ps = appState.patrolState;
  if (!ps || !ps.detections) return;

  ps.detections.forEach(d => {
    if (d.map_x == null || d.map_y == null) return;
    const p = worldToScreen(d.map_x, d.map_y);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#d1495b";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("P", p.x, p.y);
  });
}

async function boot() {
  resizeCanvas();
  updateRouteList();
  await refreshStatus();
  await refreshMap();
  await refreshOverlay();
  refreshCameraFrame();
  await refreshPatrolState();
  setInterval(refreshStatus, 1000);
  setInterval(refreshOverlay, 700);
  setInterval(refreshMap, 3000);
  setInterval(refreshDetectionStats, 1000);
  setInterval(refreshPatrolState, 500);

  document.getElementById("generatePatrolRoute").addEventListener("click", handleGenerateRoute);
  document.getElementById("startPatrol").addEventListener("click", handleStartPatrol);
  document.getElementById("stopPatrol").addEventListener("click", handleStopPatrol);
  document.getElementById("closePatrolReport").addEventListener("click", () => {
    document.getElementById("patrolReport").style.display = "none";
  });
}

boot();
