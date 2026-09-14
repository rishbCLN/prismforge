/**
 * HazardMesh — Context-Aware Construction Safety Intelligence Frontend Logic
 */

class HazardMeshApp {
  constructor() {
    this.currentScenario = null;
    this.currentFrameIdx = 0;
    this.totalFrames = 0;
    this.fps = 25.0;
    this.isPlaying = false;
    this.playbackInterval = null;
    this.activeModel = "v3_learned";
    this.cachedFrames = [];
    this.selectedWorkerId = null;

    // Domain state
    this.currentDomain = "warehouse"; // Default to Warehouse Material Handling
    this.domainWarehouseBtn = document.getElementById("domainWarehouseBtn");
    this.domainConstructionBtn = document.getElementById("domainConstructionBtn");
    this.warehouseTelemetryGrid = document.getElementById("warehouseTelemetryGrid");
    this.constructionTelemetryGrid = document.getElementById("constructionTelemetryGrid");
    this.telemetryPanelTitle = document.getElementById("telemetryPanelTitle");

    // Warehouse Telemetry elements
    this.whDropVal = document.getElementById("whDropVal");
    this.whThrowVal = document.getElementById("whThrowVal");
    this.whDragVal = document.getElementById("whDragVal");
    this.whJerkVal = document.getElementById("whJerkVal");
    this.whZoneVal = document.getElementById("whZoneVal");

    // Assistant Chat elements
    this.chatMessagesStream = document.getElementById("chatMessagesStream");
    this.chatInputField = document.getElementById("chatInputField");
    this.chatSendBtn = document.getElementById("chatSendBtn");

    // DOM Elements
    this.videoEl = document.getElementById("constructionVideo");
    this.canvasEl = document.getElementById("overlayCanvas");
    this.ctx = this.canvasEl ? this.canvasEl.getContext("2d") : null;
    this.scenarioSelect = document.getElementById("scenarioSelect");
    this.playBtn = document.getElementById("playPauseBtn");
    this.progressBarFill = document.getElementById("progressBarFill");
    this.frameCounter = document.getElementById("frameCounter");

    // Telemetry & Risk elements
    this.riskScoreVal = document.getElementById("riskScoreVal");
    this.riskGaugeCircle = document.getElementById("riskGaugeCircle");
    this.severityBadge = document.getElementById("severityBadge");
    this.priorityBadge = document.getElementById("priorityBadge");
    this.workerIdTag = document.getElementById("workerIdTag");
    this.workerHelmetVal = document.getElementById("workerHelmetVal");
    this.workerVestVal = document.getElementById("workerVestVal");
    this.workerProxVal = document.getElementById("workerProxVal");
    this.workerDurationVal = document.getElementById("workerDurationVal");
    this.factorsList = document.getElementById("factorsList");

    this.init();
  }

  async init() {
    this.setupTabs();
    this.setupDomainSwitcher();
    this.setupSupervisorChat();
    this.setupModelSelector();
    this.setupControls();
    this.setupIngestionStudio();
    await this.loadScenarios();
    await this.loadBenchmarkHub();
    await this.loadIngestionStudio();
  }

  setupTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        tabBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const target = btn.getAttribute("data-tab");
        document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
        const targetPane = document.getElementById(target);
        if (targetPane) targetPane.classList.add("active");
      });
    });
  }

  setupDomainSwitcher() {
    if (this.domainWarehouseBtn) {
      this.domainWarehouseBtn.addEventListener("click", async () => {
        if (this.currentDomain === "warehouse") return;
        this.currentDomain = "warehouse";
        this.domainWarehouseBtn.classList.add("active");
        if (this.domainConstructionBtn) this.domainConstructionBtn.classList.remove("active");
        if (this.warehouseTelemetryGrid) this.warehouseTelemetryGrid.style.display = "grid";
        if (this.constructionTelemetryGrid) this.constructionTelemetryGrid.style.display = "none";
        if (this.telemetryPanelTitle) this.telemetryPanelTitle.innerHTML = 'Warehouse Handling Kinematics &bull; <span id="workerIdTag" style="color:var(--color-cyan);">Carton #102</span>';
        await this.loadScenarios();
        await this.loadBenchmarkHub();
      });
    }

    if (this.domainConstructionBtn) {
      this.domainConstructionBtn.addEventListener("click", async () => {
        if (this.currentDomain === "construction") return;
        this.currentDomain = "construction";
        this.domainConstructionBtn.classList.add("active");
        if (this.domainWarehouseBtn) this.domainWarehouseBtn.classList.remove("active");
        if (this.warehouseTelemetryGrid) this.warehouseTelemetryGrid.style.display = "none";
        if (this.constructionTelemetryGrid) this.constructionTelemetryGrid.style.display = "grid";
        if (this.telemetryPanelTitle) this.telemetryPanelTitle.innerHTML = 'Worker Telemetry &bull; <span id="workerIdTag" style="color:var(--color-cyan);">#1</span>';
        await this.loadScenarios();
        await this.loadBenchmarkHub();
      });
    }
  }

  setupSupervisorChat() {
    if (!this.chatSendBtn || !this.chatInputField) return;

    const sendMessage = async () => {
      const text = this.chatInputField.value.trim();
      if (!text) return;
      this.chatInputField.value = "";

      // Append user bubble
      this.appendChatBubble(text, "user");

      try {
        const res = await fetch("/api/assistant/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        this.appendChatBubble(data.response || "No response received.", "assistant");
      } catch (err) {
        this.appendChatBubble("Error communicating with AI Assistant API.", "assistant");
      }
    };

    this.chatSendBtn.addEventListener("click", sendMessage);
    this.chatInputField.addEventListener("keydown", (e) => {
      if (e.key === "Enter") sendMessage();
    });

    // Handle prompt chip clicks
    document.querySelectorAll(".prompt-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        const promptText = chip.getAttribute("data-prompt");
        if (promptText) {
          this.chatInputField.value = promptText;
          sendMessage();
        }
      });
    });
  }

  appendChatBubble(text, role) {
    if (!this.chatMessagesStream) return;
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${role}`;

    // Formatting for assistant
    if (role === "assistant") {
      let formatted = text
        .replace(/### (.*?)\n/g, '<div style="font-size:15px; font-weight:700; color:#38bdf8; margin-bottom:6px;">$1</div>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n- /g, '<br>&bull; ')
        .replace(/\n/g, '<br>');
      bubble.innerHTML = formatted;
    } else {
      bubble.textContent = text;
    }

    this.chatMessagesStream.appendChild(bubble);
    this.chatMessagesStream.scrollTop = this.chatMessagesStream.scrollHeight;
  }

  setupModelSelector() {
    const pills = document.querySelectorAll(".model-pill");
    pills.forEach(pill => {
      pill.addEventListener("click", () => {
        pills.forEach(p => p.classList.remove("active"));
        pill.classList.add("active");
        this.activeModel = pill.getAttribute("data-model");
        this.updateCurrentFrameInference();
      });
    });
  }

  setupControls() {
    if (this.playBtn) {
      this.playBtn.addEventListener("click", () => this.togglePlay());
    }

    const prevBtn = document.getElementById("prevFrameBtn");
    const nextBtn = document.getElementById("nextFrameBtn");
    if (prevBtn) prevBtn.addEventListener("click", () => this.stepFrame(-1));
    if (nextBtn) nextBtn.addEventListener("click", () => this.stepFrame(1));

    const progressContainer = document.getElementById("progressBarContainer");
    if (progressContainer) {
      progressContainer.addEventListener("click", (e) => {
        const rect = progressContainer.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const targetFrame = Math.floor(ratio * (this.totalFrames - 1));
        this.seekTo(targetFrame);
      });
    }

    if (this.scenarioSelect) {
      this.scenarioSelect.addEventListener("change", (e) => {
        this.loadScenario(e.target.value);
      });
    }
  }

  async loadScenarios() {
    try {
      const url = this.currentDomain === "warehouse" ? "/api/warehouse/scenarios" : "/api/scenarios";
      const res = await fetch(url);
      const data = await res.json();
      const scenarios = data.scenarios || [];

      this.scenarioSelect.innerHTML = "";
      scenarios.forEach((s, idx) => {
        const opt = document.createElement("option");
        const clipId = s.id || s.clip_id;
        opt.value = clipId;
        const splitTag = (s.split || "train").toUpperCase();
        const sevTag = s.risk_level || s.severity || "INFO";
        const desc = s.behavior ? `${s.behavior.replace(/_/g, " ")} [${sevTag}]` : `${clipId} [${sevTag}]`;
        opt.textContent = `[${splitTag}] ${clipId} — ${desc}`;
        this.scenarioSelect.appendChild(opt);
      });

      if (scenarios.length > 0) {
        const first = scenarios[0];
        const clipId = first.id || first.clip_id;
        this.scenarioSelect.value = clipId;
        await this.loadScenario(clipId);
      }
    } catch (err) {
      console.error("Failed to load scenarios:", err);
    }
  }

  async loadScenario(clipId) {
    this.pause();
    try {
      const url = this.currentDomain === "warehouse"
        ? `/api/warehouse/scenario/${clipId}/frames`
        : `/api/scenario/${clipId}/frames`;
      const res = await fetch(url);
      const data = await res.json();
      this.currentScenario = clipId;
      this.totalFrames = data.total_frames || 100;
      this.fps = data.fps || 25.0;
      this.cachedFrames = data.features || [];
      this.cachedTracks = data.tracks || [];

      // Update video source
      this.videoEl.src = `/videos/${clipId}.mp4`;
      this.videoEl.currentTime = 0;
      this.currentFrameIdx = 0;

      // Adjust canvas resolution to match video
      this.canvasEl.width = 640;
      this.canvasEl.height = 360;

      this.renderFrame(0);
    } catch (err) {
      console.error(`Failed to load scenario ${clipId}:`, err);
    }
  }

  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  play() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    if (this.playBtn) this.playBtn.textContent = "Pause";

    const frameDurationMs = 1000.0 / this.fps;
    this.playbackInterval = setInterval(() => {
      let nextFrame = this.currentFrameIdx + 1;
      if (nextFrame >= this.totalFrames) {
        nextFrame = 0; // Loop
      }
      this.seekTo(nextFrame);
    }, frameDurationMs);
  }

  pause() {
    this.isPlaying = false;
    if (this.playBtn) this.playBtn.textContent = "Play";
    if (this.playbackInterval) {
      clearInterval(this.playbackInterval);
      this.playbackInterval = null;
    }
  }

  stepFrame(step) {
    this.pause();
    let target = this.currentFrameIdx + step;
    if (target < 0) target = 0;
    if (target >= this.totalFrames) target = this.totalFrames - 1;
    this.seekTo(target);
  }

  seekTo(frameIdx) {
    this.currentFrameIdx = frameIdx;
    if (this.videoEl && this.totalFrames > 0) {
      this.videoEl.currentTime = frameIdx / this.fps;
    }
    this.renderFrame(frameIdx);
  }

  renderFrame(frameIdx) {
    // 1. Update timeline UI
    if (this.progressBarFill && this.totalFrames > 0) {
      const pct = (frameIdx / (this.totalFrames - 1)) * 100.0;
      this.progressBarFill.style.width = `${pct}%`;
    }
    if (this.frameCounter) {
      const t = (frameIdx / this.fps).toFixed(2);
      this.frameCounter.textContent = `Frame ${frameIdx + 1} / ${this.totalFrames} (${t}s)`;
    }

    // 2. Draw canvas overlays
    this.drawOverlays(frameIdx);

    // 3. Update active worker inference
    this.updateCurrentFrameInference();
  }

  drawOverlays(frameIdx) {
    if (!this.ctx) return;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);

    const trackFrame = this.cachedTracks[frameIdx];
    if (!trackFrame) return;

    if (this.currentDomain === "warehouse") {
      // 1. Draw Staging Area Polygon
      ctx.strokeStyle = "rgba(20, 220, 245, 0.9)";
      ctx.lineWidth = 2;
      ctx.strokeRect(40, 180, 220, 150);
      ctx.fillStyle = "rgba(20, 220, 245, 0.08)";
      ctx.fillRect(40, 180, 220, 150);
      ctx.fillStyle = "#38bdf8";
      ctx.font = "bold 10px 'JetBrains Mono', monospace";
      ctx.fillText("SAFE STAGING ZONE", 48, 196);

      // 2. Draw Carton and Operator BBoxes
      if (trackFrame.carton) {
        const c = trackFrame.carton;
        const bw = c.w || 38;
        const bh = c.h || 30;
        const bx = c.x - bw / 2;
        const by = c.y - bh / 2;

        const k = trackFrame.kinematics || {};
        const isHazard = (k.drop_velocity > 0.25 || k.throw_velocity > 0.35 || k.drag_velocity > 0.2 || k.outside_staging);
        const boxStroke = isHazard ? "#ef4444" : "#10b981";

        ctx.lineWidth = 2;
        ctx.strokeStyle = boxStroke;
        ctx.strokeRect(bx, by, bw, bh);

        ctx.fillStyle = "rgba(10, 14, 23, 0.85)";
        ctx.fillRect(bx, by - 16, Math.max(70, bw), 16);
        ctx.fillStyle = boxStroke;
        ctx.font = "bold 10px 'Outfit', sans-serif";
        const tag = isHazard ? "BOX #102 [RISK]" : "BOX #102 [SAFE]";
        ctx.fillText(tag, bx + 4, by - 4);
      }
      return;
    }

    // A. Draw Machinery Danger Perimeter (Construction)
    if (trackFrame.machinery && trackFrame.machinery.length > 0) {
      trackFrame.machinery.forEach(m => {
        const [mx, my] = m.center;
        const radius = 110;

        // Pulsing danger circle
        ctx.beginPath();
        ctx.arc(mx, my, radius, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(239, 68, 68, 0.12)";
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 6]);
        ctx.strokeStyle = "rgba(239, 68, 68, 0.7)";
        ctx.stroke();
        ctx.setLineDash([]); // Reset dash

        // Machine Tag
        ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
        ctx.fillRect(mx - 45, my - radius - 18, 90, 18);
        ctx.strokeStyle = "#f59e0b";
        ctx.strokeRect(mx - 45, my - radius - 18, 90, 18);
        ctx.fillStyle = "#fbbf24";
        ctx.font = "bold 10px 'JetBrains Mono', monospace";
        ctx.fillText("DANGER ZONE", mx - 38, my - radius - 5);
      });
    }

    // B. Draw Worker Bounding Boxes & Trajectories
    const workers = trackFrame.workers || [];
    workers.forEach(w => {
      const [bx1, by1, bx2, by2] = w.bbox;
      const wid = w.track_id;
      const isSelected = (this.selectedWorkerId === null || this.selectedWorkerId === wid);

      const ppe = w.associated_ppe || {};
      const hasViolation = ppe.helmet_missing || ppe.vest_missing;

      // Color coding based on violation & machine proximity
      let strokeColor = "#10b981"; // Green (Safe)
      if (w.associated_machine && w.associated_machine.distance_px < 110) {
        strokeColor = hasViolation ? "#ef4444" : "#f59e0b"; // Red if violating, Amber if close but PPE safe
      } else if (hasViolation) {
        strokeColor = "#38bdf8"; // Cyan/Blue for safe walkway violation
      }

      ctx.lineWidth = 2;
      ctx.strokeStyle = strokeColor;
      ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);

      // Header Tag
      ctx.fillStyle = "rgba(10, 14, 23, 0.85)";
      ctx.fillRect(bx1, by1 - 20, Math.max(70, bx2 - bx1), 20);
      ctx.fillStyle = strokeColor;
      ctx.font = "bold 11px 'Outfit', sans-serif";
      ctx.fillText(`Worker #${wid}`, bx1 + 6, by1 - 6);

      // Trajectory dots
      if (w.position_history) {
        ctx.fillStyle = strokeColor;
        w.position_history.forEach(pt => {
          ctx.beginPath();
          ctx.arc(pt[0], pt[1], 2, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    });
  }

  async updateCurrentFrameInference() {
    if (!this.cachedFrames || this.cachedFrames.length === 0) return;

    if (this.currentDomain === "warehouse") {
      const sample = this.cachedFrames[this.currentFrameIdx] || this.cachedFrames.find(s => s.frame_idx === this.currentFrameIdx);
      if (!sample) return;

      const feats = sample.features || [];
      if (feats.length === 16) {
        // Update Warehouse Gauges
        if (this.whDropVal) {
          const dVal = (feats[0] * 3.5).toFixed(1);
          this.whDropVal.textContent = `${dVal} m/s`;
          this.whDropVal.className = feats[0] > 0.25 ? "t-value violation" : "t-value compliant";
        }
        if (this.whThrowVal) {
          const tVal = (feats[2] * 4.2).toFixed(1);
          this.whThrowVal.textContent = `${tVal} m/s`;
          this.whThrowVal.className = feats[2] > 0.35 ? "t-value violation" : "t-value compliant";
        }
        if (this.whDragVal) {
          const isDragging = feats[5] > 0.20;
          this.whDragVal.textContent = isDragging ? "ACTIVE (ON FLOOR)" : "NONE";
          this.whDragVal.className = isDragging ? "t-value violation" : "t-value compliant";
        }
        if (this.whJerkVal) {
          const isJerk = feats[7] > 0.40;
          this.whJerkVal.textContent = isJerk ? "EXCESSIVE SHOCK" : "CONTROLLED";
          this.whJerkVal.className = isJerk ? "t-value violation" : "t-value compliant";
        }
        if (this.whZoneVal) {
          const isOutside = feats[9] > 0.5;
          this.whZoneVal.textContent = isOutside ? "OUTSIDE DESIGNATED BAY (BREACH)" : "INSIDE SAFE STAGING ZONE";
          this.whZoneVal.className = isOutside ? "t-value violation" : "t-value compliant";
        }

        // Call Warehouse Predict API
        try {
          const res = await fetch("/api/warehouse/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model_version: this.activeModel,
              features: feats
            })
          });
          const data = await res.json();
          this.updateRiskDisplay(data.prediction);
        } catch (err) {
          console.error("Warehouse predict error:", err);
        }
      }
      return;
    }

    // Construction Domain Inference
    const frameSamples = this.cachedFrames.filter(s => s.frame_id === this.currentFrameIdx);
    if (!frameSamples || frameSamples.length === 0) return;

    // Pick target worker
    const sample = (this.selectedWorkerId !== null)
      ? (frameSamples.find(s => s.worker_id === this.selectedWorkerId) || frameSamples[0])
      : frameSamples[0];

    const f = sample.features;

    // Update Telemetry Panel
    if (this.workerIdTag) this.workerIdTag.textContent = `#${sample.worker_id}`;
    if (this.workerHelmetVal) {
      if (f.helmet_missing > 0.5) {
        this.workerHelmetVal.textContent = "MISSING";
        this.workerHelmetVal.className = "t-value violation";
      } else {
        this.workerHelmetVal.textContent = "PRESENT";
        this.workerHelmetVal.className = "t-value compliant";
      }
    }
    if (this.workerVestVal) {
      if (f.vest_missing > 0.5) {
        this.workerVestVal.textContent = "MISSING";
        this.workerVestVal.className = "t-value violation";
      } else {
        this.workerVestVal.textContent = "PRESENT";
        this.workerVestVal.className = "t-value compliant";
      }
    }
    if (this.workerProxVal) {
      const proxSev = f.machine_proximity_severity;
      if (proxSev > 0.5) {
        this.workerProxVal.textContent = `CRITICAL (${(f.normalized_worker_machine_dist * 100).toFixed(0)}%)`;
        this.workerProxVal.className = "t-value violation";
      } else if (proxSev > 0.2) {
        this.workerProxVal.textContent = `PROXIMITY (${(f.normalized_worker_machine_dist * 100).toFixed(0)}%)`;
        this.workerProxVal.className = "t-value";
      } else {
        this.workerProxVal.textContent = "CLEAR";
        this.workerProxVal.className = "t-value compliant";
      }
    }
    if (this.workerDurationVal) {
      this.workerDurationVal.textContent = `${f.violation_duration.toFixed(1)} s`;
    }

    // Call Model Prediction API for active model
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_version: this.activeModel,
          features: f
        })
      });
      const data = await res.json();
      const pred = data.prediction;

      this.updateRiskDisplay(pred);
    } catch (err) {
      console.error("Prediction API error:", err);
    }
  }

  updateRiskDisplay(pred) {
    if (!pred) return;
    const score = pred.risk_score || 0.0;
    const severity = pred.severity || "LOW";
    const priority = pred.priority || pred.intervention_priority || "P3_INFORMATIONAL";
    const factors = pred.primary_factors || pred.contributing_factors || [];

    // Circular gauge offset: circumference = 2 * PI * 45 = ~283
    const maxOffset = 283;
    const currentOffset = maxOffset - (score * maxOffset);
    if (this.riskGaugeCircle) {
      this.riskGaugeCircle.style.strokeDashoffset = currentOffset;
      // Gauge color
      if (severity === "CRITICAL" || severity === "HIGH") this.riskGaugeCircle.style.stroke = "var(--color-crimson)";
      else if (severity === "MEDIUM") this.riskGaugeCircle.style.stroke = "var(--color-amber)";
      else if (severity === "LOW") this.riskGaugeCircle.style.stroke = "var(--color-blue)";
      else this.riskGaugeCircle.style.stroke = "var(--color-green)";
    }

    if (this.riskScoreVal) {
      this.riskScoreVal.textContent = score.toFixed(2);
    }

    // Badges
    if (this.severityBadge) {
      this.severityBadge.textContent = severity;
      this.severityBadge.className = `severity-badge severity-${severity}`;
    }
    if (this.priorityBadge) {
      this.priorityBadge.textContent = priority.replace("priority-", "");
      this.priorityBadge.className = `priority-badge priority-${priority}`;
    }

    // Why Factors
    if (this.factorsList) {
      this.factorsList.innerHTML = "";
      if (factors.length === 0) {
        const chip = document.createElement("div");
        chip.className = "factor-chip";
        chip.textContent = "Normal compliant operation";
        this.factorsList.appendChild(chip);
      } else {
        factors.forEach(factorText => {
          const chip = document.createElement("div");
          const isCritical = (factorText.includes("CRITICAL") || factorText.includes("HIGH") || factorText.includes("DROP") || factorText.includes("THROW") || factorText.includes("Critical"));
          chip.className = "factor-chip" + (isCritical ? " critical" : "");
          chip.textContent = factorText.replace(/_/g, " ");
          this.factorsList.appendChild(chip);
        });
      }
    }
  }

  async loadBenchmarkHub() {
    try {
      const tbody = document.getElementById("benchmarkTableBody");
      const chartImg = document.getElementById("benchmarkChartImg");
      const chartCounter = document.getElementById("chartCounter");

      if (this.currentDomain === "warehouse") {
        if (chartImg) chartImg.src = "/reports/warehouse_improvement_chart.png";
        if (chartCounter) chartCounter.textContent = "DamageMesh Held-Out Benchmark (N=200)";

        const bRes = await fetch("/api/warehouse/benchmark");
        const bData = await bRes.json();
        if (tbody) {
          tbody.innerHTML = "";
          const vOrder = ["V0_Baseline", "V1_Context", "V2_Temporal", "V3_Learned"];
          const vLabels = {
            "V0_Baseline": "V0 Baseline (Heuristic)",
            "V1_Context": "V1 Context (Spatial)",
            "V2_Temporal": "V2 Temporal (Persistence)",
            "V3_Learned": "V3 Learned (PyTorch MLP)"
          };

          vOrder.forEach(vKey => {
            const m = bData[vKey];
            if (!m) return;
            const tr = document.createElement("tr");
            if (vKey === "V3_Learned") tr.className = "highlight-v3";
            const pillClass = vKey.toLowerCase().replace("warehouse_", "");
            tr.innerHTML = `
              <td><span class="pill-version pill-${pillClass}">${vLabels[vKey] || vKey}</span></td>
              <td><strong>${m.macro_f1.toFixed(4)}</strong></td>
              <td>${(m.high_risk_recall * 100).toFixed(1)}%</td>
              <td>${(m.false_alarm_rate * 100).toFixed(1)}%</td>
              <td><strong>${m.asymmetric_damage_cost.toFixed(1)}</strong> (5x FN + 1x FP)</td>
            `;
            tbody.appendChild(tr);
          });
        }
      } else {
        if (chartImg) chartImg.src = "/reports/improvement_chart.png";
        if (chartCounter) chartCounter.textContent = "HazardMesh Held-Out Benchmark (N=366)";

        const bRes = await fetch("/api/benchmark");
        const bData = await bRes.json();
        if (tbody && bData.models) {
          tbody.innerHTML = "";
          bData.models.forEach(m => {
            const tr = document.createElement("tr");
            if (m.version === "v3_learned") tr.className = "highlight-v3";
            const pillClass = m.version.replace("_baseline", "").replace("_context", "").replace("_temporal", "").replace("_learned", "");
            tr.innerHTML = `
              <td><span class="pill-version pill-${pillClass}">${m.display_name}</span></td>
              <td><strong>${m.macro_f1.toFixed(4)}</strong></td>
              <td>${(m.high_risk_recall * 100).toFixed(1)}%</td>
              <td>${(m.false_positive_rate * 100).toFixed(1)}%</td>
              <td>${m.weighted_hazard_error.toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
          });
        }
      }

      // 2. Failure Analysis
      const fRes = await fetch("/api/failure_analysis");
      const fData = await fRes.json();
      this.renderFailureAnalysis(fData);

      // 3. PRISM Runs
      const pRes = await fetch("/api/prism_runs");
      const pData = await pRes.json();
      const pList = document.getElementById("prismRunsList");
      if (pList && pData.runs) {
        pList.innerHTML = "";
        pData.runs.forEach(r => {
          const row = document.createElement("div");
          row.className = "prism-run-row";
          row.innerHTML = `
            <div>
              <span class="run-id-tag">${r.run_id}</span>
              <span style="margin-left: 10px; font-weight: 600; color: #fff;">${r.model_version.toUpperCase()}</span>
            </div>
            <div>
              <span style="color: var(--text-dim); margin-right: 14px;">F1: ${r.metrics.macro_f1.toFixed(3)} | Err: ${r.metrics.weighted_hazard_error.toFixed(2)}</span>
              <span class="run-commit">git:${r.git_commit}</span>
            </div>
          `;
          pList.appendChild(row);
        });
      }
    } catch (err) {
      console.error("Failed to load benchmark hub:", err);
    }
  }

  renderFailureAnalysis(fData) {
    const container = document.getElementById("failureIterationsContainer");
    if (!container || !fData.iterations) return;

    const iterations = [
      { key: "v0_to_v1", title: "Iteration 1: V0 → V1", sub: "Spatial Context Reasoning", data: fData.iterations.v0_to_v1 },
      { key: "v1_to_v2", title: "Iteration 2: V1 → V2", sub: "Temporal Persistence Gating", data: fData.iterations.v1_to_v2 },
      { key: "v2_to_v3", title: "Iteration 3: V2 → V3", sub: "Learned PyTorch MLP Reasoning", data: fData.iterations.v2_to_v3 }
    ];

    container.innerHTML = "";
    iterations.forEach(it => {
      const card = document.createElement("div");
      card.className = "iteration-card";
      const res = it.data ? it.data.results : {};
      const reduction = res ? res.error_reduction : 0;
      card.innerHTML = `
        <div class="iteration-header">
          <div>
            <div class="iteration-title">${it.title}</div>
            <div style="font-size: 11px; color: var(--text-dim);">${it.sub}</div>
          </div>
          <span class="iteration-reduction">${reduction > 0 ? `-${reduction} errors` : "Calibrated"}</span>
        </div>
        <div class="iteration-body">
          <p><strong>Diagnosis:</strong> ${it.data.diagnosis}</p>
          <p><strong>Intervention:</strong> ${it.data.intervention}</p>
        </div>
      `;
      container.appendChild(card);
    });
  }

  setupIngestionStudio() {
    // 1. Search button
    const searchBtn = document.getElementById("kaggleSearchBtn");
    const searchInput = document.getElementById("kaggleSearchInput");
    if (searchBtn && searchInput) {
      searchBtn.addEventListener("click", () => this.searchKaggle(searchInput.value.trim()));
      searchInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") this.searchKaggle(searchInput.value.trim());
      });
    }

    // 2. Custom slug download button
    const customDlBtn = document.getElementById("kaggleCustomDownloadBtn");
    if (customDlBtn) {
      customDlBtn.addEventListener("click", () => {
        const slug = document.getElementById("customSlugInput").value.trim();
        const split = document.getElementById("customSlugSplit").value;
        const autoTrain = document.getElementById("kaggleAutoTrain").checked;
        if (!slug) {
          alert("Please enter a Kaggle dataset slug (e.g. mrifatrashid/construction-site-safety-video)");
          return;
        }
        this.triggerKaggleIngestion(slug, split, autoTrain);
      });
    }

    // 3. Local file dropzone
    const dropzone = document.getElementById("uploadDropzone");
    const fileInput = document.getElementById("localFileInput");
    const fileInfo = document.getElementById("selectedFileInfo");
    const uploadBtn = document.getElementById("localUploadBtn");

    if (dropzone && fileInput) {
      dropzone.addEventListener("click", () => fileInput.click());

      dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--color-cyan)";
        dropzone.style.background = "rgba(6, 182, 212, 0.1)";
      });

      dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "var(--border-accent)";
        dropzone.style.background = "rgba(15, 23, 42, 0.4)";
      });

      dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--border-accent)";
        dropzone.style.background = "rgba(15, 23, 42, 0.4)";
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          fileInput.files = e.dataTransfer.files;
          this.handleFileSelected(fileInput.files[0]);
        }
      });

      fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files.length > 0) {
          this.handleFileSelected(fileInput.files[0]);
        }
      });

      if (uploadBtn) {
        uploadBtn.addEventListener("click", () => this.triggerLocalUpload());
      }
    }
  }

  handleFileSelected(file) {
    const fileInfo = document.getElementById("selectedFileInfo");
    const uploadBtn = document.getElementById("localUploadBtn");
    if (fileInfo && uploadBtn) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
      fileInfo.textContent = `Selected: ${file.name} (${sizeMb} MB)`;
      fileInfo.style.display = "block";
      uploadBtn.disabled = false;
    }
  }

  async loadIngestionStudio() {
    await this.checkKaggleStatus();
    await this.loadCuratedDatasets();
    await this.loadManifestTable();
  }

  async checkKaggleStatus() {
    try {
      const res = await fetch("/api/ingestion/kaggle/status");
      const data = await res.json();
      const statusBadge = document.getElementById("kaggleTokenStatus");
      const displayEl = document.getElementById("kaggleTokenDisplay");

      if (data.status === "connected") {
        if (statusBadge) {
          statusBadge.textContent = "Verified";
          statusBadge.style.color = "var(--color-green)";
        }
        if (displayEl) {
          displayEl.textContent = `${data.token_prefix} (Connected)`;
        }
      } else {
        if (statusBadge) {
          statusBadge.textContent = "Error";
          statusBadge.style.color = "var(--color-crimson)";
        }
        if (displayEl) {
          displayEl.textContent = data.error || "Authentication failed";
        }
      }
    } catch (err) {
      console.error("Failed to check Kaggle status:", err);
    }
  }

  async loadCuratedDatasets() {
    try {
      const res = await fetch("/api/ingestion/curated");
      const data = await res.json();
      const list = document.getElementById("kaggleResultsList");
      if (!list || !data.datasets) return;

      list.innerHTML = "";
      data.datasets.forEach(ds => {
        const item = document.createElement("div");
        item.style.cssText = "background: rgba(15, 23, 42, 0.5); padding: 10px 12px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); display: flex; justify-content: space-between; align-items: center;";
        item.innerHTML = `
          <div>
            <div style="font-weight: 600; font-size: 13px; color: #fff;">${ds.title}</div>
            <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--color-cyan);">${ds.slug}</div>
            <div style="font-size: 11px; color: var(--text-dim); margin-top: 2px;">${ds.description}</div>
          </div>
          <div style="text-align: right; margin-left: 12px; min-width: 100px;">
            <span style="font-size: 11px; color: var(--text-muted); display: block; margin-bottom: 4px;">${ds.size}</span>
            <button class="ctrl-btn primary" style="font-size: 11px; padding: 4px 10px;" onclick="window.app.triggerKaggleIngestion('${ds.slug}', 'train', true)">Quick Ingest</button>
          </div>
        `;
        list.appendChild(item);
      });
    } catch (err) {
      console.error("Failed to load curated datasets:", err);
    }
  }

  async searchKaggle(query) {
    if (!query) return;
    const titleEl = document.getElementById("kaggleResultsTitle");
    const list = document.getElementById("kaggleResultsList");
    if (titleEl) titleEl.textContent = `Search Results for "${query}":`;
    if (list) list.innerHTML = `<div style="font-size:12px; color:var(--text-dim); padding:10px;">Searching Kaggle...</div>`;

    try {
      const res = await fetch(`/api/ingestion/kaggle/search?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      if (!list) return;
      list.innerHTML = "";

      if (!data.results || data.results.length === 0) {
        list.innerHTML = `<div style="font-size:12px; color:var(--text-muted); padding:10px;">No datasets found for "${query}".</div>`;
        return;
      }

      data.results.forEach(ds => {
        const item = document.createElement("div");
        item.style.cssText = "background: rgba(15, 23, 42, 0.5); padding: 10px 12px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); display: flex; justify-content: space-between; align-items: center;";
        item.innerHTML = `
          <div>
            <div style="font-weight: 600; font-size: 13px; color: #fff;">${ds.title}</div>
            <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--color-cyan);">${ds.slug}</div>
            <div style="font-size: 11px; color: var(--text-dim); margin-top: 2px;">Downloads: ${ds.downloads} &bull; Votes: ${ds.votes}</div>
          </div>
          <div style="text-align: right; margin-left: 12px; min-width: 100px;">
            <span style="font-size: 11px; color: var(--text-muted); display: block; margin-bottom: 4px;">${ds.size_human}</span>
            <button class="ctrl-btn primary" style="font-size: 11px; padding: 4px 10px;" onclick="window.app.triggerKaggleIngestion('${ds.slug}', 'train', true)">Ingest</button>
          </div>
        `;
        list.appendChild(item);
      });
    } catch (err) {
      if (list) list.innerHTML = `<div style="font-size:12px; color:var(--color-crimson); padding:10px;">Search failed: ${err.message}</div>`;
    }
  }

  async triggerKaggleIngestion(datasetSlug, targetSplit, autoTrain) {
    this.showIngestionProgress(`Downloading Kaggle dataset: ${datasetSlug}...`);
    this.appendIngestionStep(`[Kaggle API] Authenticating with KGAT token...`);
    this.appendIngestionStep(`[Download] Fetching archive for ${datasetSlug}...`);

    try {
      const res = await fetch("/api/ingestion/kaggle/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_slug: datasetSlug,
          target_split: targetSplit,
          auto_train: autoTrain
        })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Ingestion request failed");
      }

      const result = await res.json();
      this.appendIngestionStep(`[Perception] YOLOv8 detection & PPE inspection complete.`);
      this.appendIngestionStep(`[Tracking] Multi-object spatio-temporal tracking finalized.`);
      this.appendIngestionStep(`[Features] Spatio-temporal features extracted & registered.`);
      if (autoTrain) {
        this.appendIngestionStep(`[Model] V3 PyTorch model successfully retrained.`);
      }
      this.finishIngestionProgress(`Ingestion Successful! Registered ${result.clips_count} clip(s).`);

      // Refresh scenario dropdown and manifest table
      await this.loadScenarios();
      await this.loadManifestTable();
      await this.loadBenchmarkHub();
    } catch (err) {
      this.finishIngestionProgress(`Error: ${err.message}`, true);
    }
  }

  async triggerLocalUpload() {
    const fileInput = document.getElementById("localFileInput");
    if (!fileInput.files || fileInput.files.length === 0) return;

    const file = fileInput.files[0];
    const split = document.getElementById("localUploadSplit").value;
    const autoTrain = document.getElementById("localAutoTrain").checked;

    this.showIngestionProgress(`Uploading & processing ${file.name}...`);
    this.appendIngestionStep(`[Upload] Uploading local file (${(file.size / (1024 * 1024)).toFixed(2)} MB)...`);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_split", split);
    formData.append("auto_train", autoTrain);

    try {
      const res = await fetch("/api/ingestion/local/upload", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Upload failed");
      }

      const result = await res.json();
      this.appendIngestionStep(`[Video Normalizer] Standardized to 640x360 @ 25 FPS & chunked.`);
      this.appendIngestionStep(`[Perception] Extracted detections & PPE states.`);
      this.appendIngestionStep(`[Features] Generated spatio-temporal tabular records.`);
      if (autoTrain) {
        this.appendIngestionStep(`[Model] Retrained V3 PyTorch model.`);
      }
      this.finishIngestionProgress(`Upload complete! Ingested ${result.clips_count} clip(s).`);

      // Refresh scenario dropdown and manifest table
      await this.loadScenarios();
      await this.loadManifestTable();
      await this.loadBenchmarkHub();
    } catch (err) {
      this.finishIngestionProgress(`Error: ${err.message}`, true);
    }
  }

  showIngestionProgress(initialMsg) {
    const box = document.getElementById("ingestionProgressBox");
    const msg = document.getElementById("ingestionStatusMsg");
    const list = document.getElementById("ingestionStepList");
    if (box) box.style.display = "block";
    if (msg) {
      msg.textContent = initialMsg;
      msg.style.color = "var(--color-cyan)";
    }
    if (list) list.innerHTML = "";
  }

  appendIngestionStep(stepText) {
    const list = document.getElementById("ingestionStepList");
    if (list) {
      const line = document.createElement("div");
      line.textContent = `> ${stepText}`;
      list.appendChild(line);
    }
  }

  finishIngestionProgress(finalMsg, isError = false) {
    const msg = document.getElementById("ingestionStatusMsg");
    if (msg) {
      msg.textContent = finalMsg;
      msg.style.color = isError ? "var(--color-crimson)" : "var(--color-green)";
    }
  }

  async loadManifestTable() {
    try {
      const res = await fetch("/api/ingestion/manifest");
      const data = await res.json();
      const tbody = document.getElementById("manifestTableBody");
      const countEl = document.getElementById("manifestClipCount");

      if (!tbody || !data.clips) return;

      tbody.innerHTML = "";
      if (countEl) countEl.textContent = `${data.clips.length} Clips`;

      data.clips.forEach(c => {
        const tr = document.createElement("tr");
        const splitClass = c.split === "train" ? "pill-v3" : (c.split === "val" ? "pill-v1" : "pill-v2");
        const sourceLabel = c.source || "synthetic";
        tr.innerHTML = `
          <td style="font-weight:600; color:#fff;">${c.clip_id}</td>
          <td><span class="pill-version ${splitClass}">${c.split.toUpperCase()}</span></td>
          <td>${c.frames}</td>
          <td style="color:var(--color-cyan); font-family:'JetBrains Mono',monospace; font-size:11px;">${sourceLabel}</td>
          <td style="color:var(--text-muted); font-size:11px;">${c.video_file}</td>
        `;
        tbody.appendChild(tr);
      });
    } catch (err) {
      console.error("Failed to load manifest table:", err);
    }
  }
}

// Instantiate on load
document.addEventListener("DOMContentLoaded", () => {
  window.app = new HazardMeshApp();
});
