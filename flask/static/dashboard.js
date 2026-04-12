const appState = {
  map: null,
  overlay: null,
  status: null,
  draftRoute: [],
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

async function boot() {
  resizeCanvas();
  updateRouteList();
  await refreshStatus();
  await refreshMap();
  await refreshOverlay();
  refreshCameraFrame();
  setInterval(refreshStatus, 1000);
  setInterval(refreshOverlay, 700);
  setInterval(refreshMap, 3000);
}

boot();
