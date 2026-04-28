const REGION_COLORS = [
  { fill: "rgba(74, 144, 217, 0.20)", stroke: "rgba(74, 144, 217, 0.85)" },
  { fill: "rgba(217, 130, 43, 0.20)", stroke: "rgba(217, 130, 43, 0.85)" },
  { fill: "rgba(71, 184, 129, 0.20)", stroke: "rgba(71, 184, 129, 0.85)" },
  { fill: "rgba(217, 74, 106, 0.20)", stroke: "rgba(217, 74, 106, 0.85)" },
  { fill: "rgba(138, 99, 210, 0.20)", stroke: "rgba(138, 99, 210, 0.85)" },
  { fill: "rgba(217, 184, 74, 0.20)", stroke: "rgba(217, 184, 74, 0.85)" },
  { fill: "rgba(74, 184, 217, 0.20)", stroke: "rgba(74, 184, 217, 0.85)" },
  { fill: "rgba(217, 106, 74, 0.20)", stroke: "rgba(217, 106, 74, 0.85)" },
  { fill: "rgba(106, 138, 217, 0.20)", stroke: "rgba(106, 138, 217, 0.85)" },
  { fill: "rgba(74, 217, 138, 0.20)", stroke: "rgba(74, 217, 138, 0.85)" },
  { fill: "rgba(184, 74, 217, 0.20)", stroke: "rgba(184, 74, 217, 0.85)" },
  { fill: "rgba(217, 74, 74, 0.20)", stroke: "rgba(217, 74, 74, 0.85)" },
  { fill: "rgba(74, 217, 184, 0.20)", stroke: "rgba(74, 217, 184, 0.85)" },
  { fill: "rgba(138, 106, 74, 0.20)", stroke: "rgba(138, 106, 74, 0.85)" },
  { fill: "rgba(106, 217, 74, 0.20)", stroke: "rgba(106, 217, 74, 0.85)" },
];

const appState = {
  map: null,
  overlay: null,
  status: null,
  draftRoute: [],
  patrolRoute: [],
  patrolState: null,
  routeMode: false,
  regions: [],
  regionColors: {},
  regionHighlight: null,
  selectedRegion: null,
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
  document.getElementById("mapFps").textContent = status.map_fps > 0
    ? `${status.map_fps.toFixed(1)} FPS`
    : "";
  document.getElementById("cameraStatus").textContent = status.camera_available
    ? `相机在线 ${new Date(status.camera_stamp * 1000).toLocaleTimeString()}`
    : "相机未连接";
  document.getElementById("cameraFps").textContent = status.camera_fps > 0
    ? `${status.camera_fps.toFixed(1)} FPS`
    : "";

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
  appState.robotRegion = data.robot_region;
  updateStatusPanel(data);
}

async function refreshMap() {
  try {
    const seq = appState.map ? appState.map.seq : 0;
    const data = await fetchJSON(`/api/map?seq=${seq}`);
    if (data.ok && data.changed) {
      appState.map = data;
      fitMapToView();
      draw();
    }
  } catch (_err) {
  }
}

async function refreshOverlay() {
  const data = await fetchJSON("/api/overlay");
  if (!data.ok) return;
  // preserve pose from fast pose endpoint if available
  const currentPose = appState.overlay && appState.overlay._fastPose;
  appState.overlay = data;
  if (currentPose) appState.overlay.pose = currentPose;
  draw();
}

async function refreshPose() {
  try {
    const data = await fetchJSON("/api/pose");
    if (!data.ok) return;
    if (!appState.overlay) appState.overlay = {};
    appState.overlay._fastPose = data.pose;
    appState.overlay.pose = data.pose;
    if (data.speed && appState.status) appState.status.speed = data.speed;
    draw();
  } catch (_err) {
  }
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
  drawRegions();

  if (appState.overlay) {
    drawPolyline(appState.overlay.plan, "#355f8d", 3);
    drawPolyline(appState.overlay.trajectory, "#db8f2f", 2);
    drawPolyline(appState.overlay.current_route, "#5b4ab8", 2);
    drawWaypoints(appState.overlay.current_route, appState.overlay.current_waypoint_index, "#5b4ab8");
    drawRobot(appState.overlay.pose);
  }

  drawPolyline(appState.draftRoute, "#1d5c4e", 2);
  drawWaypoints(appState.draftRoute, -1, "#1d5c4e");

  // highlight robot region from live status
  if (appState.robotRegion && !appState.regionHighlight) {
    appState.regionHighlight = appState.robotRegion;
  }

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

// ---- region segmentation ----

async function refreshRegions() {
  try {
    const data = await fetchJSON("/api/map/regions");
    if (data.ok) {
      const changed = JSON.stringify(data.regions) !== JSON.stringify(appState.regions);
      appState.regions = data.regions || [];
      appState.robotRegion = data.robot_region;
      if (changed) {
        assignRegionColors();
        updateRegionList();
        draw();
      }
    }
  } catch (_err) {
    // ignore
  }
}

function assignRegionColors() {
  const colors = {};
  appState.regions.forEach((reg, i) => {
    colors[reg.id] = REGION_COLORS[i % REGION_COLORS.length];
  });
  appState.regionColors = colors;
}

function drawRegions() {
  if (!appState.map || !appState.regions.length) return;

  appState.regions.forEach(reg => {
    const colors = appState.regionColors[reg.id];
    if (!colors) return;
    const boundary = reg.boundary || [];
    const holes = reg.holes || [];
    if (boundary.length < 3) return;

    ctx.beginPath();

    // outer boundary
    boundary.forEach((pt, i) => {
      const p = worldToScreen(pt.x, pt.y);
      ctx[i === 0 ? "moveTo" : "lineTo"](p.x, p.y);
    });
    ctx.closePath();

    // holes (reverse winding for evenodd rule)
    holes.forEach(hole => {
      if (hole.length < 3) return;
      // trace in reverse order to ensure opposite winding
      for (let i = hole.length - 1; i >= 0; i--) {
        const p = worldToScreen(hole[i].x, hole[i].y);
        ctx[i === hole.length - 1 ? "moveTo" : "lineTo"](p.x, p.y);
      }
      ctx.closePath();
    });

    // fill
    ctx.fillStyle = colors.fill;
    ctx.fill("evenodd");

    // stroke boundary
    ctx.strokeStyle = colors.stroke;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.stroke();

    // draw label at center
    const navTargets = reg.nav_targets || {};
    const centerPt = navTargets.center;
    if (centerPt) {
      const sp = worldToScreen(centerPt.x, centerPt.y);
      const label = reg.label || reg.id;
      ctx.fillStyle = "#142b26";
      ctx.font = "bold 13px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      // text background for readability
      const metrics = ctx.measureText(label);
      const tw = metrics.width + 12;
      const th = 24;
      ctx.fillStyle = "rgba(255, 250, 242, 0.88)";
      const rr = 6;
      const rx = sp.x - tw / 2, ry = sp.y - th / 2;
      ctx.beginPath();
      ctx.moveTo(rx + rr, ry);
      ctx.lineTo(rx + tw - rr, ry);
      ctx.arcTo(rx + tw, ry, rx + tw, ry + rr, rr);
      ctx.lineTo(rx + tw, ry + th - rr);
      ctx.arcTo(rx + tw, ry + th, rx + tw - rr, ry + th, rr);
      ctx.lineTo(rx + rr, ry + th);
      ctx.arcTo(rx, ry + th, rx, ry + th - rr, rr);
      ctx.lineTo(rx, ry + rr);
      ctx.arcTo(rx, ry, rx + rr, ry, rr);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = colors.stroke;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = colors.stroke;
      ctx.font = "bold 13px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, sp.x, sp.y);
    }

    // highlight robot region
    if (reg.id === appState.regionHighlight) {
      ctx.strokeStyle = "#d1495b";
      ctx.lineWidth = 4;
      ctx.setLineDash([6, 4]);
      // re-trace outer boundary for highlight
      ctx.beginPath();
      boundary.forEach((pt, i) => {
        const p = worldToScreen(pt.x, pt.y);
        ctx[i === 0 ? "moveTo" : "lineTo"](p.x, p.y);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    }
  });
}

function updateRegionList() {
  const container = document.getElementById("regionList");
  if (!appState.regions.length) {
    container.innerHTML = '<span class="hint">点击"分割地图"自动识别房间和走廊</span>';
    document.getElementById("saveRegionsBtn").disabled = true;
    document.getElementById("clearRegionsBtn").disabled = true;
    return;
  }

  document.getElementById("saveRegionsBtn").disabled = false;
  document.getElementById("clearRegionsBtn").disabled = false;

  const typeLabels = { room: "房间", corridor: "走廊" };
  container.innerHTML = appState.regions.map(reg => {
    const colors = appState.regionColors[reg.id];
    const colorHex = colors ? colors.stroke : "#888";
    const typeLabel = typeLabels[reg.type] || reg.type;
    const areaText = reg.area_sqm ? `${reg.area_sqm.toFixed(1)} m²` : "";
    const connects = (reg.connects_to || []).length
      ? `<div class="region-connects">↳ 连接: ${reg.connects_to.join(", ")}</div>`
      : "";
    const highlightCls = reg.id === appState.regionHighlight ? "region-highlight" : "";
    return `<div class="region-item ${highlightCls}" data-region-id="${reg.id}">
      <span class="region-color-dot" style="background:${colorHex}"></span>
      <span class="region-label" data-region-id="${reg.id}">${reg.label}</span>
      <span class="region-type-badge">${typeLabel}</span>
      <span class="region-area">${areaText}</span>
      ${connects}
    </div>`;
  }).join("");

  // click label to rename
  container.querySelectorAll(".region-label").forEach(el => {
    el.addEventListener("click", () => startRename(el));
  });
}

function startRename(labelEl) {
  const regionId = labelEl.dataset.regionId;
  const currentLabel = labelEl.textContent;
  const input = document.createElement("input");
  input.type = "text";
  input.value = currentLabel;
  input.className = "region-rename-input";
  input.maxLength = 32;

  const finishRename = async () => {
    const newLabel = input.value.trim();
    if (newLabel && newLabel !== currentLabel) {
      try {
        const res = await fetchJSON("/api/map/regions/rename", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: regionId, label: newLabel }),
        });
        if (res.ok) {
          await refreshRegions();
          draw();
        }
      } catch (_err) {
        // ignore
      }
    }
    // restore label display
    labelEl.style.display = "";
    input.replaceWith(labelEl);
  };

  input.addEventListener("blur", finishRename);
  input.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); input.blur(); }
    if (e.key === "Escape") { input.value = currentLabel; input.blur(); }
  });

  labelEl.style.display = "none";
  labelEl.parentNode.insertBefore(input, labelEl.nextSibling);
  input.focus();
  input.select();
}

async function handleSegmentMap() {
  const btn = document.getElementById("segmentMapBtn");
  btn.disabled = true;
  btn.textContent = "分割中...";
  try {
    const res = await fetchJSON("/api/map/segment", { method: "POST" });
    if (!res.ok) {
      document.getElementById("commandResult").textContent = `分割失败: ${res.error}`;
    } else {
      document.getElementById("commandResult").textContent =
        `分割完成: ${res.regions.length} 个区域，机器人在 ${res.robot_region || "未知区域"}`;
    }
  } catch (_err) {
    document.getElementById("commandResult").textContent = "分割请求失败";
  }
  btn.disabled = false;
  btn.textContent = "分割地图";
  await refreshRegions();
  draw();
}

async function handleSaveRegions() {
  try {
    const res = await fetchJSON("/api/map/regions/save", { method: "POST" });
    if (res.ok) {
      document.getElementById("commandResult").textContent =
        `区域已保存到: ${res.filepath} (${res.region_count} 个区域)`;
    } else {
      document.getElementById("commandResult").textContent = `保存失败: ${res.error}`;
    }
  } catch (_err) {
    document.getElementById("commandResult").textContent = "保存请求失败";
  }
}

async function handleClearRegions() {
  await fetchJSON("/api/map/regions/clear", { method: "POST" });
  appState.regions = [];
  appState.regionColors = {};
  appState.regionHighlight = null;
  updateRegionList();
  draw();
  document.getElementById("commandResult").textContent = "区域数据已清除";
}

async function boot() {
  resizeCanvas();
  updateRouteList();
  await refreshStatus();
  await refreshMap();
  await refreshOverlay();
  refreshCameraFrame();
  await refreshPatrolState();
  await refreshRegions();
  setInterval(refreshStatus, 1000);
  setInterval(refreshOverlay, 500);
  setInterval(refreshPose, 500);
  setInterval(refreshMap, 1000);
  setInterval(refreshDetectionStats, 1000);
  setInterval(refreshPatrolState, 500);
  setInterval(refreshRegions, 2000);

  document.getElementById("generatePatrolRoute").addEventListener("click", handleGenerateRoute);
  document.getElementById("startPatrol").addEventListener("click", handleStartPatrol);
  document.getElementById("stopPatrol").addEventListener("click", handleStopPatrol);
  document.getElementById("closePatrolReport").addEventListener("click", () => {
    document.getElementById("patrolReport").style.display = "none";
  });
  document.getElementById("segmentMapBtn").addEventListener("click", handleSegmentMap);
  document.getElementById("saveRegionsBtn").addEventListener("click", handleSaveRegions);
  document.getElementById("clearRegionsBtn").addEventListener("click", handleClearRegions);

  // canvas click → select region (only when not in route mode)
  canvas.addEventListener("click", (event) => {
    if (appState.routeMode) return;
    const rect = canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;

    // hit-test regions in reverse draw order (topmost first)
    appState.regionHighlight = null;
    for (let i = appState.regions.length - 1; i >= 0; i--) {
      const reg = appState.regions[i];
      const boundary = reg.boundary || [];
      if (boundary.length < 3) continue;

      // convert boundary to screen coords
      const screenPts = boundary.map(pt => worldToScreen(pt.x, pt.y));
      if (isPointInPolygon(mx, my, screenPts)) {
        appState.regionHighlight = reg.id;
        // If holes, check that point is not in a hole
        const holes = reg.holes || [];
        let inHole = false;
        for (const hole of holes) {
          if (hole.length < 3) continue;
          const holePts = hole.map(pt => worldToScreen(pt.x, pt.y));
          if (isPointInPolygon(mx, my, holePts)) {
            inHole = true;
            break;
          }
        }
        if (!inHole) break;
        appState.regionHighlight = null;
      }
    }
    updateRegionList();
    draw();
  });
}

// point-in-polygon ray casting
function isPointInPolygon(px, py, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    if (((yi > py) !== (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

boot();
