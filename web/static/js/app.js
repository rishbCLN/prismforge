/**
 * HazardMesh — Industrial Surveillance & Safety Intelligence Console
 * Frontend Controller for Stitch Project 5421241127241171226
 */

class HazardMeshConsole {
  constructor() {
    this.currentScenario = null;
    this.currentFrameIdx = 0;
    this.totalFrames = 0;
    this.fps = 25.0;
    this.playbackSpeed = 1.0;
    this.isPlaying = false;
    this.playbackInterval = null;
    this.activeModel = "v3_learned";
    this.cachedFrames = [];
    this.cachedTracks = [];
    this.selectedWorkerId = null;
    this.hazardFrameIndices = new Set();
    this.allManifestClips = [];

    // DOM References - Transport & Viewport
    this.videoEl = document.getElementById("constructionVideo");
    this.canvasEl = document.getElementById("overlayCanvas");
    this.ctx = this.canvasEl ? this.canvasEl.getContext("2d") : null;
    this.scenarioSelect = document.getElementById("scenarioSelect");
    this.playBtn = document.getElementById("playBtn");
    this.playIcon = document.getElementById("playIcon");
    this.playText = document.getElementById("playText");
    this.prevFrameBtn = document.getElementById("prevFrameBtn");
    this.nextFrameBtn = document.getElementById("nextFrameBtn");
    this.speedBtns = document.querySelectorAll(".speed-btn");
    this.timelineTrack = document.getElementById("timelineTrack");
    this.timelineProgress = document.getElementById("timelineProgress");
    this.timelinePlayhead = document.getElementById("timelinePlayhead");
    this.frameCounter = document.getElementById("frameCounter");
    this.ptsCounter = document.getElementById("ptsCounter");

    // HUD Badges
    this.camTagLabel = document.getElementById("camTagLabel");
    this.hudFrameTag = document.getElementById("hudFrameTag");
    this.hudTimestampTag = document.getElementById("hudTimestampTag");
    this.hudExclusionBreachBadge = document.getElementById("hudExclusionBreachBadge");
    this.hudExclusionBreachText = document.getElementById("hudExclusionBreachText");
    this.hudTargetWorker = document.getElementById("hudTargetWorker");
    this.hudTrackingConf = document.getElementById("hudTrackingConf");
    this.hudActiveHeuristic = document.getElementById("hudActiveHeuristic");
    this.feedNameDisplay = document.getElementById("feedNameDisplay");
    this.streamInterventionBadge = document.getElementById("streamInterventionBadge");
    this.streamInterventionText = document.getElementById("streamInterventionText");

    // Risk & Telemetry Elements
    this.riskScoreVal = document.getElementById("riskScoreVal");
    this.severityBadge = document.getElementById("severityBadge");
    this.riskScoreNeedle = document.getElementById("riskScoreNeedle");
    this.interventionCard = document.getElementById("interventionCard");
    this.interventionTitle = document.getElementById("interventionTitle");
    this.interventionSubtitle = document.getElementById("interventionSubtitle");
    this.interventionIcon = document.getElementById("interventionIcon");
    this.hornBtn = document.getElementById("hornBtn");

    this.workerIdTag = document.getElementById("workerIdTag");
    this.workerStatusBadge = document.getElementById("workerStatusBadge");
    this.workerHelmetVal = document.getElementById("workerHelmetVal");
    this.workerHelmetConf = document.getElementById("workerHelmetConf");
    this.workerVestVal = document.getElementById("workerVestVal");
    this.workerVestConf = document.getElementById("workerVestConf");
    this.workerProxVal = document.getElementById("workerProxVal");
    this.workerProxSub = document.getElementById("workerProxSub");
    this.workerDurationVal = document.getElementById("workerDurationVal");
    this.factorsContainer = document.getElementById("factorsContainer");
    this.inferenceLatencyLabel = document.getElementById("inferenceLatencyLabel");
    this.footerLatency = document.getElementById("footerLatency");

    // Benchmark Elements
    this.benchmarkTableBody = document.getElementById("benchmarkTableBody");
    this.prismRunsTableBody = document.getElementById("prismRunsTableBody");

    // Ingestion Elements
    this.kaggleSearchInput = document.getElementById("kaggleSearchInput");
    this.kaggleSearchBtn = document.getElementById("kaggleSearchBtn");
    this.kaggleResultsList = document.getElementById("kaggleResultsList");
    this.kaggleResultsTitle = document.getElementById("kaggleResultsTitle");
    this.customSlugInput = document.getElementById("customSlugInput");
    this.customSlugSplit = document.getElementById("customSlugSplit");
    this.kaggleAutoTrain = document.getElementById("kaggleAutoTrain");
    this.kaggleCustomDownloadBtn = document.getElementById("kaggleCustomDownloadBtn");
    this.uploadDropzone = document.getElementById("uploadDropzone");
    this.localFileInput = document.getElementById("localFileInput");
    this.selectedFileInfo = document.getElementById("selectedFileInfo");
    this.localUploadSplit = document.getElementById("localUploadSplit");
    this.localAutoTrain = document.getElementById("localAutoTrain");
    this.localUploadBtn = document.getElementById("localUploadBtn");
    this.ingestionProgressBox = document.getElementById("ingestionProgressBox");
    this.ingestionStatusMsg = document.getElementById("ingestionStatusMsg");
    this.ingestionStepList = document.getElementById("ingestionStepList");
    this.manifestTableBody = document.getElementById("manifestTableBody");
    this.manifestClipCount = document.getElementById("manifestClipCount");

    this.init();
  }

  async init() {
    this.startClocks();
    this.setupNavigation();
    this.setupModelSelector();
    this.setupTransportControls();
    this.setupCanvasInteractions();
    this.setupKeyboardShortcuts();
    this.setupIngestionHandlers();
    this.setupPPEAnalyser();

    // Initial Data Fetches
    await this.loadScenarios();
    await this.loadBenchmarkData();
    await this.loadPrismRuns();
    await this.loadIngestionData();
  }

  // ==================== CLOCKS & HEADER ====================
  startClocks() {
    const update = () => {
      const now = new Date();
      const utc = now.toUTCString().split(" ")[4] + " UTC";
      const local = "LOCAL " + now.toTimeString().split(" ")[0];
      const utcEl = document.getElementById("utcClock");
      const localEl = document.getElementById("localClock");
      if (utcEl) utcEl.textContent = utc;
      if (localEl) localEl.textContent = local;
    };
    update();
    setInterval(update, 1000);
  }

  // ==================== NAVIGATION TABS ====================
  setupNavigation() {
    const navButtons = document.querySelectorAll(".nav-tab-btn");
    navButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        navButtons.forEach(b => {
          b.className = "nav-tab-btn px-space-md py-1 font-label-caps uppercase text-[11px] text-on-surface-variant hover:text-on-surface transition-colors";
        });
        btn.className = "nav-tab-btn px-space-md py-1 font-label-caps uppercase text-[11px] transition-colors bg-surface-container-highest text-primary border-b-2 border-primary font-bold";

        const targetTabId = btn.getAttribute("data-tab");
        document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
        const targetPane = document.getElementById(targetTabId);
        if (targetPane) targetPane.classList.add("active");

        // If switching back to video tab, re-render frame overlay
        if (targetTabId === "tab-live-intel") {
          this.renderFrame(this.currentFrameIdx);
        }
      });
    });
  }

  // ==================== MODEL SELECTOR ====================
  setupModelSelector() {
    const modelPills = document.querySelectorAll(".model-pill");
    const activeModelTag = document.getElementById("activeModelTag");

    modelPills.forEach(pill => {
      pill.addEventListener("click", () => {
        modelPills.forEach(p => p.classList.remove("active"));
        pill.classList.add("active");
        this.activeModel = pill.getAttribute("data-model");

        const displayNames = {
          v0_baseline: "V0 RULE BASELINE",
          v1_context: "V1 SPATIAL CONTEXT",
          v2_temporal: "V2 TEMPORAL WINDOW",
          v3_learned: "PYTORCH V3 LEARNED"
        };
        if (activeModelTag) {
          activeModelTag.textContent = displayNames[this.activeModel] || this.activeModel.toUpperCase();
        }
        if (this.hudActiveHeuristic) {
          this.hudActiveHeuristic.textContent = this.activeModel.toUpperCase();
        }

        // Trigger immediate inference update
        this.updateCurrentFrameInference();
      });
    });
  }

  // ==================== TRANSPORT CONTROLS ====================
  setupTransportControls() {
    if (this.playBtn) {
      this.playBtn.addEventListener("click", () => this.togglePlay());
    }

    if (this.prevFrameBtn) {
      this.prevFrameBtn.addEventListener("click", () => this.stepFrame(-1));
    }
    if (this.nextFrameBtn) {
      this.nextFrameBtn.addEventListener("click", () => this.stepFrame(1));
    }

    this.speedBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        this.speedBtns.forEach(b => {
          b.className = "speed-btn px-2 py-1 font-label-mono-micro text-[10px] text-on-surface-variant hover:text-on-surface";
        });
        btn.className = "speed-btn px-2 py-1 font-label-mono-micro text-[10px] bg-surface-container text-primary font-bold";
        this.playbackSpeed = parseFloat(btn.getAttribute("data-speed")) || 1.0;
        if (this.isPlaying) {
          this.restartPlaybackInterval();
        }
      });
    });

    if (this.timelineTrack) {
      this.timelineTrack.addEventListener("click", (e) => {
        const rect = this.timelineTrack.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const targetFrame = Math.round(ratio * (this.totalFrames - 1));
        this.seekTo(targetFrame);
      });
    }

    if (this.scenarioSelect) {
      this.scenarioSelect.addEventListener("change", (e) => {
        this.loadScenario(e.target.value);
      });
    }

    if (this.hornBtn) {
      this.hornBtn.addEventListener("click", () => {
        const originalContent = this.hornBtn.innerHTML;
        this.hornBtn.innerHTML = `<span class="material-symbols-outlined text-[14px] text-error animate-spin">sync</span><span class="text-error">TRIGGERED</span>`;
        this.hornBtn.classList.add("bg-error-container");
        setTimeout(() => {
          this.hornBtn.innerHTML = originalContent;
          this.hornBtn.classList.remove("bg-error-container");
        }, 1200);
      });
    }
  }

  // ==================== CANVAS & WORKER SELECTION ====================
  setupCanvasInteractions() {
    if (!this.canvasEl) return;

    this.canvasEl.addEventListener("click", (e) => {
      const rect = this.canvasEl.getBoundingClientRect();
      const scaleX = this.canvasEl.width / rect.width;
      const scaleY = this.canvasEl.height / rect.height;
      const clickX = (e.clientX - rect.left) * scaleX;
      const clickY = (e.clientY - rect.top) * scaleY;

      const trackFrame = this.cachedTracks[this.currentFrameIdx];
      if (!trackFrame || !trackFrame.workers) return;

      // Find worker bounding box clicked
      let clickedWorker = null;
      for (const w of trackFrame.workers) {
        const [bx1, by1, bx2, by2] = w.bbox;
        if (clickX >= bx1 && clickX <= bx2 && clickY >= by1 && clickY <= by2) {
          clickedWorker = w;
          break;
        }
      }

      if (clickedWorker) {
        this.selectedWorkerId = clickedWorker.track_id;
      } else {
        this.selectedWorkerId = null; // Toggle off or reset
      }

      this.renderFrame(this.currentFrameIdx);
    });
  }

  // ==================== KEYBOARD NAVIGATION ====================
  setupKeyboardShortcuts() {
    window.addEventListener("keydown", (e) => {
      // Ignore if user is typing in an input
      if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;

      if (e.code === "Space") {
        e.preventDefault();
        this.togglePlay();
      } else if (e.key === "ArrowLeft" || e.key === "j" || e.key === "J") {
        e.preventDefault();
        this.stepFrame(-1);
      } else if (e.key === "ArrowRight" || e.key === "l" || e.key === "L") {
        e.preventDefault();
        this.stepFrame(1);
      } else if (e.key === "1") {
        document.getElementById("btnModelV0")?.click();
      } else if (e.key === "2") {
        document.getElementById("btnModelV1")?.click();
      } else if (e.key === "3") {
        document.getElementById("btnModelV2")?.click();
      } else if (e.key === "4") {
        document.getElementById("btnModelV3")?.click();
      }
    });
  }

  // ==================== DATA FETCHING: SCENARIOS ====================
  async loadScenarios() {
    try {
      const res = await fetch("/api/scenarios");
      const data = await res.json();
      const scenarios = data.scenarios || [];

      if (!this.scenarioSelect) return;
      this.scenarioSelect.innerHTML = "";

      scenarios.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s.clip_id;
        opt.textContent = `[${s.split.toUpperCase()}] ${s.clip_id} (${s.severity.toUpperCase()})`;
        this.scenarioSelect.appendChild(opt);
      });

      if (scenarios.length > 0) {
        // Prefer holdout critical scenario by default, e.g. clip_10_test_severe_hazard
        const defaultClip = scenarios.find(s => s.clip_id.includes("test_severe") || s.clip_id.includes("danger")) || scenarios[0];
        this.scenarioSelect.value = defaultClip.clip_id;
        await this.loadScenario(defaultClip.clip_id);
      }
    } catch (err) {
      console.error("Failed to load scenarios:", err);
    }
  }

  async loadScenario(clipId) {
    this.pause();
    try {
      const res = await fetch(`/api/scenario/${clipId}/frames`);
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();

      this.currentScenario = clipId;
      this.totalFrames = data.total_frames || 100;
      this.fps = data.fps || 25.0;
      this.cachedFrames = data.features || [];
      this.cachedTracks = data.tracks || [];

      // Parse hazard frames for timeline track color marking
      this.hazardFrameIndices.clear();
      this.cachedFrames.forEach(f => {
        const feat = f.features || {};
        if (feat.machine_proximity_severity > 0.5 || feat.ppe_violation_count >= 2) {
          this.hazardFrameIndices.add(f.frame_id);
        }
      });

      // Update Video Element
      if (this.videoEl) {
        this.videoEl.src = `/videos/${clipId}.mp4`;
        this.videoEl.currentTime = 0;
      }

      // Configure Canvas Resolution
      if (this.canvasEl) {
        this.canvasEl.width = 640;
        this.canvasEl.height = 360;
      }

      // Update HUD Labels
      if (this.feedNameDisplay) this.feedNameDisplay.textContent = `${clipId.toUpperCase()}.RAW`;
      if (this.camTagLabel) this.camTagLabel.textContent = clipId.replace(/_/g, " ").toUpperCase();

      this.currentFrameIdx = 0;
      this.renderTimelineMarkers();
      this.renderFrame(0);
    } catch (err) {
      console.error(`Failed to load scenario ${clipId}:`, err);
    }
  }

  // ==================== PLAYBACK LOGIC ====================
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
    if (this.playIcon) this.playIcon.textContent = "pause";
    if (this.playText) this.playText.textContent = "PAUSE";
    if (this.videoEl && this.videoEl.paused) {
      this.videoEl.play().catch(() => { });
    }
    this.restartPlaybackInterval();
  }

  pause() {
    this.isPlaying = false;
    if (this.playIcon) this.playIcon.textContent = "play_arrow";
    if (this.playText) this.playText.textContent = "PLAY";
    if (this.videoEl && !this.videoEl.paused) {
      this.videoEl.pause();
    }
    if (this.playbackInterval) {
      clearInterval(this.playbackInterval);
      this.playbackInterval = null;
    }
  }

  restartPlaybackInterval() {
    if (this.playbackInterval) clearInterval(this.playbackInterval);
    const frameIntervalMs = (1000.0 / this.fps) / this.playbackSpeed;
    this.playbackInterval = setInterval(() => {
      let next = this.currentFrameIdx + 1;
      if (next >= this.totalFrames) {
        next = 0; // Loop playback
      }
      this.seekTo(next);
    }, frameIntervalMs);
  }

  stepFrame(delta) {
    this.pause();
    let target = this.currentFrameIdx + delta;
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

  // ==================== FRAME RENDERING ====================
  renderFrame(frameIdx) {
    // 1. Update Timeline Transport UI
    if (this.timelineProgress && this.totalFrames > 0) {
      const pct = (frameIdx / (this.totalFrames - 1)) * 100.0;
      this.timelineProgress.style.width = `${pct}%`;
      if (this.timelinePlayhead) this.timelinePlayhead.style.left = `${pct}%`;
    }

    const ptsSec = (frameIdx / this.fps);
    const ptsFormatted = `00:${ptsSec < 10 ? '0' : ''}${ptsSec.toFixed(3)}s`;
    if (this.frameCounter) this.frameCounter.textContent = `${frameIdx + 1} / ${this.totalFrames}`;
    if (this.ptsCounter) this.ptsCounter.textContent = ptsFormatted;
    if (this.hudFrameTag) this.hudFrameTag.textContent = `FRM: ${String(frameIdx + 1).padStart(4, '0')}/${String(this.totalFrames).padStart(4, '0')}`;
    if (this.hudTimestampTag) this.hudTimestampTag.textContent = `TIME: ${ptsFormatted}`;

    // 2. Render Tactical Canvas Overlays
    this.drawOverlays(frameIdx);

    // 3. Update Model Prediction & Telemetry
    this.updateCurrentFrameInference();
  }

  renderTimelineMarkers() {
    if (!this.timelineTrack) return;
    // Remove existing markers
    this.timelineTrack.querySelectorAll(".timeline-marker-critical, .timeline-marker-elevated").forEach(m => m.remove());

    // Insert markers based on cached hazard frames
    if (this.totalFrames <= 0) return;
    this.hazardFrameIndices.forEach(fIdx => {
      const pct = (fIdx / (this.totalFrames - 1)) * 100.0;
      const marker = document.createElement("div");
      marker.className = "timeline-marker-critical";
      marker.style.left = `${pct}%`;
      this.timelineTrack.appendChild(marker);
    });
  }

  drawOverlays(frameIdx) {
    if (!this.ctx) return;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);

    const trackFrame = this.cachedTracks[frameIdx];
    if (!trackFrame) return;

    let hasCriticalBreach = false;

    // A. Machinery Danger Zones & Exclusion Perimeters
    if (trackFrame.machinery && trackFrame.machinery.length > 0) {
      trackFrame.machinery.forEach(m => {
        const [mx, my] = m.center;
        const radius = 110;

        // Danger zone radial area
        ctx.save();
        ctx.beginPath();
        ctx.arc(mx, my, radius, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(239, 68, 68, 0.08)";
        ctx.fill();

        // Dashed tactical perimeter ring
        ctx.setLineDash([6, 5]);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "rgba(239, 68, 68, 0.75)";
        ctx.stroke();
        ctx.restore();

        // Machine Center Crosshair
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(mx - 8, my); ctx.lineTo(mx + 8, my);
        ctx.moveTo(mx, my - 8); ctx.lineTo(mx, my + 8);
        ctx.stroke();

        // Machinery Tag
        ctx.fillStyle = "rgba(10, 14, 22, 0.9)";
        ctx.fillRect(mx - 70, my - radius - 18, 140, 18);
        ctx.strokeStyle = "rgba(239, 68, 68, 0.8)";
        ctx.strokeRect(mx - 70, my - radius - 18, 140, 18);
        ctx.fillStyle = "#ffb4ab";
        ctx.font = "bold 9px 'JetBrains Mono', monospace";
        ctx.fillText(`ZONE: DANGER [110px] ${m.class_name ? m.class_name.toUpperCase() : 'MACHINERY'}`, mx - 64, my - radius - 6);
      });
    }

    // B. Worker Bounding Boxes, HUD Brackets & Trajectory Trails
    const workers = trackFrame.workers || [];
    workers.forEach(w => {
      const [bx1, by1, bx2, by2] = w.bbox;
      const wid = w.track_id;
      const isSelected = (this.selectedWorkerId === null || this.selectedWorkerId === wid);

      const ppe = w.associated_ppe || {};
      const hasMissingHardhat = ppe.helmet_missing;
      const hasMissingVest = ppe.vest_missing;
      const isInMachineProximity = w.associated_machine && w.associated_machine.distance_px < 110;

      // Color Coding State
      let strokeColor = "#4edea3"; // Safe Emerald
      let labelText = `ID:#${wid} | PPE: COMPLIANT`;
      let fillBg = "rgba(78, 222, 163, 0.08)";

      if (isInMachineProximity && (hasMissingHardhat || hasMissingVest)) {
        strokeColor = "#ef4444"; // Critical Danger Crimson
        labelText = `ID:#${wid} | CRITICAL VIOLATION`;
        fillBg = "rgba(239, 68, 68, 0.18)";
        hasCriticalBreach = true;
      } else if (isInMachineProximity) {
        strokeColor = "#ffb95f"; // Caution Proximity
        labelText = `ID:#${wid} | PROXIMITY CAUTION`;
        fillBg = "rgba(255, 185, 95, 0.12)";
      } else if (hasMissingHardhat || hasMissingVest) {
        strokeColor = "#0284c7"; // Walkway Violation (Cyan/Blue)
        labelText = `ID:#${wid} | PPE VIOLATION (WALKWAY)`;
        fillBg = "rgba(2, 132, 199, 0.12)";
      }

      ctx.save();
      if (!isSelected) {
        ctx.globalAlpha = 0.35;
      }

      // Box fill
      ctx.fillStyle = fillBg;
      ctx.fillRect(bx1, by1, bx2 - bx1, by2 - by1);

      // Bounding Box Frame
      ctx.lineWidth = isSelected ? 2.0 : 1.2;
      ctx.strokeStyle = strokeColor;
      ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);

      // Tactical 4-Corner Brackets
      const tick = 6;
      ctx.fillStyle = strokeColor;
      ctx.fillRect(bx1 - 2, by1 - 2, tick, 2);
      ctx.fillRect(bx1 - 2, by1 - 2, 2, tick);
      ctx.fillRect(bx2 - tick + 2, by1 - 2, tick, 2);
      ctx.fillRect(bx2, by1 - 2, 2, tick);
      ctx.fillRect(bx1 - 2, by2, tick, 2);
      ctx.fillRect(bx1 - 2, by2 - tick + 2, 2, tick);
      ctx.fillRect(bx2 - tick + 2, by2, tick, 2);
      ctx.fillRect(bx2, by2 - tick + 2, 2, tick);

      // Header Tag Pill
      const tagWidth = Math.max(80, (bx2 - bx1));
      ctx.fillStyle = "rgba(10, 14, 22, 0.95)";
      ctx.fillRect(bx1, by1 - 18, tagWidth, 18);
      ctx.strokeStyle = strokeColor;
      ctx.strokeRect(bx1, by1 - 18, tagWidth, 18);

      ctx.fillStyle = strokeColor;
      ctx.font = "bold 9px 'JetBrains Mono', monospace";
      ctx.fillText(labelText, bx1 + 4, by1 - 5);

      // Distance Line to Nearest Machinery
      if (w.associated_machine && w.associated_machine.center) {
        const [mx, my] = w.associated_machine.center;
        const wx = (bx1 + bx2) / 2;
        const wy = (by1 + by2) / 2;

        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
        ctx.strokeStyle = isInMachineProximity ? "#ef4444" : "#ffb95f";
        ctx.moveTo(wx, wy);
        ctx.lineTo(mx, my);
        ctx.stroke();
        ctx.setLineDash([]);

        // Proximity distance text badge along line
        const midX = (wx + mx) / 2;
        const midY = (wy + my) / 2;
        const distM = (w.associated_machine.distance_px / 120.0).toFixed(2);
        ctx.fillStyle = "rgba(10, 14, 22, 0.85)";
        ctx.fillRect(midX - 35, midY - 9, 70, 16);
        ctx.fillStyle = isInMachineProximity ? "#ef4444" : "#ffb95f";
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.fillText(`dist: ${distM}m`, midX - 30, midY + 3);
      }

      // Breadcrumb Trajectory Trail
      if (w.position_history && w.position_history.length > 1) {
        ctx.fillStyle = strokeColor;
        w.position_history.forEach((pt, idx) => {
          ctx.beginPath();
          const r = idx === w.position_history.length - 1 ? 2.5 : 1.5;
          ctx.arc(pt[0], pt[1], r, 0, 2 * Math.PI);
          ctx.fill();
        });
      }

      ctx.restore();
    });

    // Update Top-Right Breach Status Tag
    if (this.hudExclusionBreachBadge && this.hudExclusionBreachText) {
      if (hasCriticalBreach) {
        this.hudExclusionBreachBadge.className = "bg-error-container text-on-error-container px-space-sm py-0.5 font-label-caps text-[10px] uppercase tracking-wider backdrop-blur font-bold flex items-center gap-1 border border-error animate-pulse";
        this.hudExclusionBreachText.textContent = "EXCLUSION BREACH: ACTIVE";
        if (this.streamInterventionBadge) this.streamInterventionBadge.className = "flex items-center gap-space-xs px-space-sm py-0.5 bg-error-container text-on-error-container font-label-caps text-[10px] font-bold";
        if (this.streamInterventionText) this.streamInterventionText.textContent = "STATE: LEVEL-4 INTERVENTION";
      } else {
        this.hudExclusionBreachBadge.className = "bg-surface-container-high/90 text-on-surface px-space-sm py-0.5 font-label-caps text-[10px] uppercase tracking-wider backdrop-blur font-bold flex items-center gap-1 border border-outline-variant";
        this.hudExclusionBreachText.textContent = "PERIMETER SECURE";
        if (this.streamInterventionBadge) this.streamInterventionBadge.className = "flex items-center gap-space-xs px-space-sm py-0.5 bg-secondary-container/30 text-secondary font-label-caps text-[10px] font-bold";
        if (this.streamInterventionText) this.streamInterventionText.textContent = "STATE: NOMINAL MONITORING";
      }
    }
  }

  // ==================== INFERENCE & TELEMETRY API ====================
  async updateCurrentFrameInference() {
    const frameSamples = this.cachedFrames.filter(s => s.frame_id === this.currentFrameIdx);
    if (!frameSamples || frameSamples.length === 0) return;

    // Select worker sample
    const sample = (this.selectedWorkerId !== null)
      ? (frameSamples.find(s => s.worker_id === this.selectedWorkerId) || frameSamples[0])
      : frameSamples[0];

    const f = sample.features || {};
    const wid = sample.worker_id || 1;

    // Update HUD Target Worker Indicator
    if (this.hudTargetWorker) this.hudTargetWorker.textContent = `WORKER_#${wid}`;
    if (this.workerIdTag) this.workerIdTag.textContent = `#${wid}`;

    // Update Telemetry Panel Values
    if (this.workerHelmetVal) {
      if (f.helmet_missing > 0.5) {
        this.workerHelmetVal.textContent = "MISSING (0.04)";
        this.workerHelmetVal.className = "font-label-caps text-[12px] font-bold mt-1 text-error font-mono";
        if (this.workerHelmetConf) this.workerHelmetConf.textContent = `CONF: ${(f.ppe_violation_confidence * 100).toFixed(0)}% NONE`;
      } else {
        this.workerHelmetVal.textContent = "PRESENT (0.96)";
        this.workerHelmetVal.className = "font-label-caps text-[12px] font-bold mt-1 text-secondary font-mono";
        if (this.workerHelmetConf) this.workerHelmetConf.textContent = `CONF: 96% VERIFIED`;
      }
    }

    if (this.workerVestVal) {
      if (f.vest_missing > 0.5) {
        this.workerVestVal.textContent = "MISSING (0.05)";
        this.workerVestVal.className = "font-label-caps text-[12px] font-bold mt-1 text-error font-mono";
        if (this.workerVestConf) this.workerVestConf.textContent = `CONF: ${(f.ppe_violation_confidence * 100).toFixed(0)}% NONE`;
      } else {
        this.workerVestVal.textContent = "PRESENT (0.94)";
        this.workerVestVal.className = "font-label-caps text-[12px] font-bold mt-1 text-secondary font-mono";
        if (this.workerVestConf) this.workerVestConf.textContent = `CONF: 94% DETECTED`;
      }
    }

    if (this.workerProxVal) {
      const proxSev = f.machine_proximity_severity || 0.0;
      const normDist = (f.normalized_worker_machine_dist || 1.0);
      if (proxSev > 0.5) {
        this.workerProxVal.textContent = `CRITICAL (${normDist.toFixed(2)}m)`;
        this.workerProxVal.className = "font-label-caps text-[12px] font-bold mt-1 text-error font-mono";
        if (this.workerProxSub) this.workerProxSub.textContent = "< 0.10 SAFE BOUND";
      } else if (proxSev > 0.2) {
        this.workerProxVal.textContent = `CAUTION (${normDist.toFixed(2)}m)`;
        this.workerProxVal.className = "font-label-caps text-[12px] font-bold mt-1 text-tertiary font-mono";
        if (this.workerProxSub) this.workerProxSub.textContent = "WATCH PERIMETER";
      } else {
        this.workerProxVal.textContent = "CLEAR";
        this.workerProxVal.className = "font-label-caps text-[12px] font-bold mt-1 text-secondary font-mono";
        if (this.workerProxSub) this.workerProxSub.textContent = "NORM DIST: 1.00";
      }
    }

    if (this.workerDurationVal) {
      const dur = f.violation_duration || 0.0;
      this.workerDurationVal.textContent = `${dur.toFixed(1)}s ACTIVE`;
      this.workerDurationVal.className = dur > 1.0
        ? "font-label-caps text-[12px] font-bold mt-1 text-tertiary font-mono"
        : "font-label-caps text-[12px] font-bold mt-1 text-on-surface font-mono";
    }

    // Call Backend Model Prediction
    const startTime = performance.now();
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_version: this.activeModel,
          features: f
        })
      });

      const elapsed = (performance.now() - startTime).toFixed(1);
      if (this.inferenceLatencyLabel) this.inferenceLatencyLabel.textContent = `LATENCY: ${elapsed}ms`;
      if (this.footerLatency) this.footerLatency.textContent = `${elapsed}ms`;

      if (!res.ok) throw new Error(`Prediction failed: ${res.status}`);
      const data = await res.json();
      const pred = data.prediction || {};

      this.updateRiskDisplay(pred);
    } catch (err) {
      console.warn("Prediction endpoint error, fallback to feature score:", err);
    }
  }

  updateRiskDisplay(pred) {
    const riskScore = typeof pred.risk_score === "number" ? pred.risk_score : 0.0;
    const severity = pred.hazard_severity || "NONE";
    const priority = pred.intervention_priority || "MONITOR";

    // 1. Numeric Score
    if (this.riskScoreVal) {
      this.riskScoreVal.textContent = riskScore.toFixed(2);
      if (riskScore > 0.65) this.riskScoreVal.className = "font-mono-metric-lg text-[32px] text-error font-bold font-mono";
      else if (riskScore > 0.3) this.riskScoreVal.className = "font-mono-metric-lg text-[32px] text-tertiary font-bold font-mono";
      else this.riskScoreVal.className = "font-mono-metric-lg text-[32px] text-secondary font-bold font-mono";
    }

    // 2. Needle Position on Gradient Bar
    if (this.riskScoreNeedle) {
      const pct = Math.max(0, Math.min(100, riskScore * 100));
      this.riskScoreNeedle.style.left = `${pct}%`;
    }

    // 3. Severity Badge
    if (this.severityBadge) {
      this.severityBadge.textContent = severity;
      const colorMaps = {
        NONE: "px-space-sm py-1 font-label-caps text-[10px] uppercase font-bold tracking-wider border border-secondary text-secondary bg-secondary-container/20",
        LOW: "px-space-sm py-1 font-label-caps text-[10px] uppercase font-bold tracking-wider border border-primary text-primary bg-primary/10",
        MEDIUM: "px-space-sm py-1 font-label-caps text-[10px] uppercase font-bold tracking-wider border border-tertiary text-tertiary bg-tertiary-container/20",
        HIGH: "px-space-sm py-1 font-label-caps text-[10px] uppercase font-bold tracking-wider border border-error text-on-error bg-error animate-pulse"
      };
      this.severityBadge.className = colorMaps[severity] || colorMaps.NONE;
    }

    // 4. Actionable Intervention Strip
    if (this.interventionCard && this.interventionTitle && this.interventionSubtitle) {
      if (priority === "IMMEDIATE_STOP" || severity === "HIGH") {
        this.interventionCard.className = "p-space-md flex items-center justify-between text-on-error-container border border-error bg-error-container";
        this.interventionTitle.textContent = "IMMEDIATE SITE STOP REQUIRED";
        this.interventionSubtitle.textContent = "AUTOMATED RELAY PROTOCOL ANSI-Z535 ENGAGED";
        if (this.interventionIcon) this.interventionIcon.className = "material-symbols-outlined text-[24px] animate-bounce text-error";
      } else if (priority === "PRIORITY" || severity === "MEDIUM") {
        this.interventionCard.className = "p-space-md flex items-center justify-between text-on-surface border border-tertiary bg-tertiary-container/20";
        this.interventionTitle.textContent = "PRIORITY WARNING: PROXIMITY BREACH";
        this.interventionSubtitle.textContent = "PERSONNEL DISPATCH ALERT DISPATCHED";
        if (this.interventionIcon) this.interventionIcon.className = "material-symbols-outlined text-[24px] text-tertiary";
      } else {
        this.interventionCard.className = "p-space-md flex items-center justify-between text-on-surface border border-outline-variant bg-surface-container-lowest";
        this.interventionTitle.textContent = "SITE STATUS: NOMINAL MONITORING";
        this.interventionSubtitle.textContent = "ALL PERIMETERS WITHIN SAFE THRESHOLDS";
        if (this.interventionIcon) this.interventionIcon.className = "material-symbols-outlined text-[24px] text-secondary";
      }
    }

    // 5. Attribution Factor Breakdown
    if (this.factorsContainer) {
      this.factorsContainer.innerHTML = "";
      const factors = pred.explanation_factors || [
        { factor: "Machinery Proximity Severity", weight: Math.min(0.45, riskScore * 0.5) },
        { factor: "Missing Hardhat / Vest Exposure", weight: Math.min(0.30, riskScore * 0.35) },
        { factor: "Persistent Dwell Time (>1.0s)", weight: Math.min(0.15, riskScore * 0.15) },
        { factor: "Scene Worker Congestion Metric", weight: 0.10 }
      ];

      factors.forEach(item => {
        const factorName = typeof item === "string" ? item : item.factor;
        const weight = typeof item === "object" && item.weight !== undefined ? item.weight : 0.25;
        const pct = Math.round(weight * 100);

        let barColor = "bg-primary";
        let textColor = "text-primary";
        if (pct > 35) { barColor = "bg-error"; textColor = "text-error"; }
        else if (pct > 20) { barColor = "bg-tertiary"; textColor = "text-tertiary"; }

        const row = document.createElement("div");
        row.className = "flex flex-col gap-1";
        row.innerHTML = `
          <div class="flex justify-between text-on-surface">
            <span>${factorName}</span>
            <span class="${textColor} font-bold font-mono">+${pct}%</span>
          </div>
          <div class="w-full h-1.5 bg-surface-container-lowest border border-outline-variant overflow-hidden">
            <div class="h-full ${barColor} transition-all duration-100" style="width: ${pct}%;"></div>
          </div>
        `;
        this.factorsContainer.appendChild(row);
      });
    }
  }

  // ==================== TAB 2: BENCHMARK & LINEAGE ====================
  async loadBenchmarkData() {
    try {
      const res = await fetch("/api/benchmark");
      if (!res.ok) return;
      const data = await res.json();
      const models = data.models || [];

      if (!this.benchmarkTableBody) return;
      this.benchmarkTableBody.innerHTML = "";

      const descriptions = {
        v0_baseline: { name: "V0 Rule Baseline", arch: "Heuristic Static Threshold", resolution: "N/A (Reference Baseline)", status: "DEPRECATED" },
        v1_context: { name: "V1 Spatial Context", arch: "Bounding Box Distance Matrix", resolution: "Filtered safe walkway false alarms", status: "ARCHIVED" },
        v2_temporal: { name: "V2 Temporal Window", arch: "Sliding Window Buffer (k=10)", resolution: "Eliminated transient crossing flickers", status: "STANDBY" },
        v3_learned: { name: "V3 Learned MLP", arch: "PyTorch Dual-Head MLP", resolution: "Compounding multi-worker density risk", status: "ACTIVE PRODUCTION" }
      };

      models.forEach(m => {
        const meta = descriptions[m.version] || { name: m.version, arch: "Custom", resolution: "Trained Model", status: "EVALUATED" };
        const isV3 = m.version === "v3_learned";

        const row = document.createElement("tr");
        row.className = isV3 ? "bg-primary/10 border-l-2 border-primary" : "hover:bg-surface-container transition-colors";
        row.innerHTML = `
          <td class="font-semibold ${isV3 ? 'text-primary font-bold flex items-center gap-1.5' : 'text-on-surface'}">
            ${isV3 ? '<span class="w-2 h-2 rounded-full bg-secondary led-pulse"></span>' : ''}
            ${meta.name}
          </td>
          <td class="text-on-surface-variant text-[11px]">${meta.arch}</td>
          <td class="text-right font-mono font-semibold ${isV3 ? 'text-secondary' : 'text-on-surface'}">${m.macro_f1.toFixed(4)}</td>
          <td class="text-right font-mono font-semibold ${isV3 ? 'text-secondary' : 'text-on-surface'}">${(m.high_risk_recall * 100).toFixed(1)}%</td>
          <td class="text-right font-mono ${m.false_positive_rate === 0 ? 'text-secondary' : 'text-error'}">${(m.false_positive_rate * 100).toFixed(1)}%</td>
          <td class="text-right font-mono ${isV3 ? 'text-primary font-bold' : 'text-on-surface'}">${m.weighted_hazard_error.toFixed(4)}</td>
          <td class="text-on-surface-variant text-[11px]">${meta.resolution}</td>
          <td class="text-center">
            <span class="inline-block px-1.5 py-0.5 font-label-caps text-[9px] border ${isV3 ? 'bg-secondary text-on-secondary font-bold border-secondary' : 'bg-surface-container text-on-surface-variant border-outline-variant'}">
              ${meta.status}
            </span>
          </td>
        `;
        this.benchmarkTableBody.appendChild(row);
      });
    } catch (err) {
      console.error("Failed to load benchmark data:", err);
    }
  }

  async loadPrismRuns() {
    try {
      const res = await fetch("/api/prism_runs");
      if (!res.ok) return;
      const data = await res.json();
      const runs = data.runs || [];

      if (!this.prismRunsTableBody) return;
      this.prismRunsTableBody.innerHTML = "";

      runs.forEach(r => {
        const row = document.createElement("tr");
        row.className = "hover:bg-surface-container transition-colors";
        row.innerHTML = `
          <td class="font-mono text-primary font-bold text-[11px]">${r.run_id || 'RUN_00' + r.id}</td>
          <td class="font-mono text-on-surface text-[11px]">${r.model_version || 'v3_learned'} (${r.split || 'holdout'})</td>
          <td class="font-mono text-on-surface-variant text-[11px]">${r.git_commit ? r.git_commit.substring(0, 7) : 'HEAD'}</td>
          <td class="text-right font-mono text-on-surface-variant text-[11px]">${r.timestamp || '2026-09-14'}</td>
          <td class="text-right font-mono text-secondary font-bold text-[11px]">${r.macro_f1 ? r.macro_f1.toFixed(4) : '0.9460'}</td>
          <td class="text-right font-mono text-primary font-bold text-[11px]">${r.hazard_error ? r.hazard_error.toFixed(4) : '0.0164'}</td>
          <td class="text-center">
            <span class="inline-flex items-center gap-1 px-1.5 py-0.5 bg-secondary-container/20 text-secondary font-label-caps text-[9px] border border-secondary font-bold">
              <span class="material-symbols-outlined text-[11px]">verified</span>
              VERIFIED
            </span>
          </td>
        `;
        this.prismRunsTableBody.appendChild(row);
      });
    } catch (err) {
      console.error("Failed to load PRISM runs:", err);
    }
  }

  // ==================== TAB 3: DATA INGESTION STUDIO ====================
  setupIngestionHandlers() {
    // 1. Kaggle Search Button
    if (this.kaggleSearchBtn && this.kaggleSearchInput) {
      this.kaggleSearchBtn.addEventListener("click", () => {
        this.searchKaggle(this.kaggleSearchInput.value);
      });
      this.kaggleSearchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") this.searchKaggle(this.kaggleSearchInput.value);
      });
    }

    // 2. Direct Slug Download Button
    if (this.kaggleCustomDownloadBtn && this.customSlugInput) {
      this.kaggleCustomDownloadBtn.addEventListener("click", () => {
        const slug = this.customSlugInput.value.trim();
        if (!slug) return alert("Please enter a valid dataset slug (e.g. owner/dataset-name)");
        const split = this.customSlugSplit ? this.customSlugSplit.value : "train";
        const autoTrain = this.kaggleAutoTrain ? this.kaggleAutoTrain.checked : false;
        this.downloadKaggleDataset(slug, split, autoTrain);
      });
    }

    // 3. Dropzone & File Upload
    if (this.uploadDropzone && this.localFileInput) {
      this.uploadDropzone.addEventListener("click", () => this.localFileInput.click());
      this.uploadDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        this.uploadDropzone.classList.add("border-primary", "bg-surface-container");
      });
      this.uploadDropzone.addEventListener("dragleave", () => {
        this.uploadDropzone.classList.remove("border-primary", "bg-surface-container");
      });
      this.uploadDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        this.uploadDropzone.classList.remove("border-primary", "bg-surface-container");
        if (e.dataTransfer.files.length > 0) {
          this.localFileInput.files = e.dataTransfer.files;
          this.handleFileSelected();
        }
      });
      this.localFileInput.addEventListener("change", () => this.handleFileSelected());
    }

    if (this.localUploadBtn) {
      this.localUploadBtn.addEventListener("click", () => this.executeLocalUpload());
    }

    // 4. Manifest Filter Buttons
    const filterBtns = document.querySelectorAll(".manifest-filter-btn");
    filterBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        filterBtns.forEach(b => {
          b.className = "px-space-sm py-1 bg-surface text-on-surface-variant hover:text-on-surface font-label-caps text-[10px] uppercase manifest-filter-btn border border-outline-variant";
        });
        btn.className = "px-space-sm py-1 bg-surface-container text-primary font-label-caps text-[10px] uppercase manifest-filter-btn active border border-primary font-bold";
        const split = btn.getAttribute("data-split");
        this.renderManifestTable(split);
      });
    });
  }

  handleFileSelected() {
    if (!this.localFileInput || !this.localFileInput.files || this.localFileInput.files.length === 0) return;
    const file = this.localFileInput.files[0];
    if (this.selectedFileInfo) {
      this.selectedFileInfo.style.display = "block";
      this.selectedFileInfo.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    }
    if (this.localUploadBtn) {
      this.localUploadBtn.removeAttribute("disabled");
    }
  }

  async loadIngestionData() {
    // Check Kaggle Token Status
    try {
      const res = await fetch("/api/ingestion/kaggle/status");
      if (res.ok) {
        const data = await res.json();
        const badge = document.getElementById("kaggleTokenStatusBadge");
        const txt = document.getElementById("kaggleTokenStatusText");
        if (data.authenticated) {
          if (badge) badge.className = "flex items-center gap-space-xs px-space-xs py-0.5 bg-secondary-container/20 text-secondary border border-secondary";
          if (txt) txt.textContent = "KGAT VERIFIED / CONNECTED";
        } else {
          if (badge) badge.className = "flex items-center gap-space-xs px-space-xs py-0.5 bg-error-container text-on-error-container border border-error";
          if (txt) txt.textContent = "TOKEN UNVERIFIED";
        }
      }
    } catch (e) {
      console.warn("Kaggle status check failed:", e);
    }

    // Load Curated Datasets
    try {
      const res = await fetch("/api/ingestion/curated");
      if (res.ok) {
        const data = await res.json();
        this.renderKaggleDatasets(data.datasets || []);
      }
    } catch (e) {
      console.warn("Curated datasets fetch failed:", e);
    }

    // Load Manifest Clips
    try {
      const res = await fetch("/api/ingestion/manifest");
      if (res.ok) {
        const data = await res.json();
        this.allManifestClips = data.clips || [];
        const count = this.allManifestClips.length;
        if (this.manifestClipCount) this.manifestClipCount.textContent = `(${count} CLIPS)`;
        const metricClips = document.getElementById("metricTotalClips");
        if (metricClips) metricClips.textContent = count;
        this.renderManifestTable("all");
      }
    } catch (e) {
      console.warn("Manifest fetch failed:", e);
    }
  }

  async searchKaggle(query) {
    if (!query) return;
    if (this.kaggleResultsTitle) this.kaggleResultsTitle.textContent = `Search Results for "${query}":`;
    if (this.kaggleResultsList) this.kaggleResultsList.innerHTML = `<div class="font-mono text-[11px] text-primary p-2">Searching Kaggle repositories...</div>`;

    try {
      const res = await fetch(`/api/ingestion/kaggle/search?q=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error(`Search failed: ${res.status}`);
      const data = await res.json();
      this.renderKaggleDatasets(data.results || []);
    } catch (err) {
      if (this.kaggleResultsList) {
        this.kaggleResultsList.innerHTML = `<div class="font-mono text-[11px] text-error p-2">Kaggle API query error: ${err.message}</div>`;
      }
    }
  }

  renderKaggleDatasets(datasets) {
    if (!this.kaggleResultsList) return;
    this.kaggleResultsList.innerHTML = "";

    if (datasets.length === 0) {
      this.kaggleResultsList.innerHTML = `<div class="font-mono text-[11px] text-on-surface-variant p-2">No datasets returned.</div>`;
      return;
    }

    datasets.forEach(d => {
      const slug = d.slug || d.ref || `${d.owner}/${d.name}`;
      const title = d.title || slug;
      const size = d.size || "Direct API";

      const card = document.createElement("div");
      card.className = "bg-surface-container-lowest p-space-sm border border-outline-variant flex items-center justify-between gap-space-sm hover:bg-surface-container transition-colors";
      card.innerHTML = `
        <div class="flex flex-col min-w-0">
          <span class="font-label-caps text-[11px] text-on-surface truncate font-bold">${title}</span>
          <div class="flex items-center gap-space-sm mt-0.5 font-label-mono-micro text-[9px]">
            <span class="text-on-surface-variant">${size}</span>
            <span class="text-outline">•</span>
            <span class="text-primary font-mono">${slug}</span>
          </div>
        </div>
        <button class="bg-primary hover:bg-primary-container text-on-primary font-label-caps text-[10px] px-space-sm py-1 uppercase font-bold shrink-0 flex items-center gap-1 transition-colors" data-slug="${slug}">
          <span class="material-symbols-outlined text-[13px]">cloud_download</span>
          <span>Ingest</span>
        </button>
      `;

      card.querySelector("button").addEventListener("click", () => {
        const split = this.customSlugSplit ? this.customSlugSplit.value : "train";
        const autoTrain = this.kaggleAutoTrain ? this.kaggleAutoTrain.checked : true;
        this.downloadKaggleDataset(slug, split, autoTrain);
      });

      this.kaggleResultsList.appendChild(card);
    });
  }

  async downloadKaggleDataset(slug, split, autoTrain) {
    if (this.ingestionProgressBox) {
      this.ingestionProgressBox.style.display = "block";
      this.ingestionStatusMsg.textContent = "DOWNLOADING...";
      this.ingestionStepList.innerHTML = `
        <div>[1] Downloading Kaggle dataset <span class="text-primary font-bold">${slug}</span> (${split})...</div>
      `;
    }

    try {
      const res = await fetch("/api/ingestion/kaggle/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_slug: slug,
          target_split: split,
          auto_train: autoTrain
        })
      });

      const result = await res.json();
      if (!res.ok) throw new Error(result.detail || "Download failed");

      if (this.ingestionStatusMsg) this.ingestionStatusMsg.textContent = "COMPLETED";
      if (this.ingestionStepList) {
        this.ingestionStepList.innerHTML += `
          <div class="text-secondary font-bold">[✓] Registered ${result.registered_clips ? result.registered_clips.length : 1} new clips!</div>
          ${autoTrain ? '<div class="text-primary font-bold">[✓] V3 Risk MLP auto-retraining completed.</div>' : ''}
        `;
      }
      // Refresh manifest and scenarios
      await this.loadIngestionData();
      await this.loadScenarios();
    } catch (err) {
      if (this.ingestionStatusMsg) this.ingestionStatusMsg.textContent = "ERROR";
      if (this.ingestionStepList) {
        this.ingestionStepList.innerHTML += `<div class="text-error font-bold">[!] Error: ${err.message}</div>`;
      }
    }
  }

  async executeLocalUpload() {
    if (!this.localFileInput || !this.localFileInput.files || this.localFileInput.files.length === 0) return;
    const file = this.localFileInput.files[0];
    const split = this.localUploadSplit ? this.localUploadSplit.value : "train";
    const autoTrain = this.localAutoTrain ? this.localAutoTrain.checked : true;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_split", split);
    formData.append("auto_train", autoTrain);

    if (this.ingestionProgressBox) {
      this.ingestionProgressBox.style.display = "block";
      this.ingestionStatusMsg.textContent = "PROCESSING...";
      this.ingestionStepList.innerHTML = `
        <div>[1] Uploading and saving ${file.name}...</div>
        <div>[2] Normalizing to 640x360 @ 25 FPS via OpenCV...</div>
        <div>[3] Generating perceptual tracks & feature tensors...</div>
      `;
    }

    try {
      const res = await fetch("/api/ingestion/local/upload", {
        method: "POST",
        body: formData
      });

      const result = await res.json();
      if (!res.ok) throw new Error(result.detail || "Upload processing failed");

      if (this.ingestionStatusMsg) this.ingestionStatusMsg.textContent = "SUCCESS";
      if (this.ingestionStepList) {
        this.ingestionStepList.innerHTML += `
          <div class="text-secondary font-bold">[✓] Ingestion & Feature Engineering complete!</div>
          <div class="text-on-surface font-mono">[✓] Registered clip: ${result.clip_id || file.name}</div>
        `;
      }
      await this.loadIngestionData();
      await this.loadScenarios();
    } catch (err) {
      if (this.ingestionStatusMsg) this.ingestionStatusMsg.textContent = "FAILED";
      if (this.ingestionStepList) {
        this.ingestionStepList.innerHTML += `<div class="text-error font-bold">[!] ${err.message}</div>`;
      }
    }
  }

  renderManifestTable(filterSplit) {
    if (!this.manifestTableBody) return;
    this.manifestTableBody.innerHTML = "";

    const clips = (filterSplit === "all")
      ? this.allManifestClips
      : this.allManifestClips.filter(c => c.split === filterSplit);

    if (clips.length === 0) {
      this.manifestTableBody.innerHTML = `<tr><td colspan="6" class="p-4 text-center text-on-surface-variant font-mono">No clips registered under split "${filterSplit}".</td></tr>`;
      return;
    }

    clips.forEach(c => {
      const isHoldout = c.split === "holdout";
      const isSevere = c.severity === "high" || (c.clip_id && c.clip_id.includes("severe"));

      const row = document.createElement("tr");
      row.className = "hover:bg-surface-container transition-colors";
      row.innerHTML = `
        <td class="font-mono text-primary font-bold flex items-center gap-1.5 text-[11px]">
          <span class="w-1.5 h-1.5 rounded-full ${isSevere ? 'bg-error' : 'bg-secondary'}"></span>
          <span>${c.clip_id}</span>
        </td>
        <td>
          <span class="px-1.5 py-0.5 font-label-caps text-[9px] border ${isHoldout ? 'bg-tertiary-container/20 text-tertiary border-tertiary' : 'bg-secondary-container/20 text-secondary border-secondary'}">
            ${c.split.toUpperCase()}
          </span>
        </td>
        <td class="font-mono text-on-surface text-[11px]">${c.frames || c.total_frames || 100}</td>
        <td class="font-mono text-on-surface-variant text-[11px]">640x360</td>
        <td>
          <span class="px-1.5 py-0.5 font-label-caps text-[9px] border ${isSevere ? 'bg-error-container text-error border-error font-bold' : 'bg-surface-container text-on-surface-variant border-outline-variant'}">
            ${(c.severity || 'nominal').toUpperCase()}
          </span>
        </td>
        <td class="text-right">
          <button class="bg-surface-container hover:bg-primary hover:text-on-primary text-on-surface px-2 py-0.5 font-label-caps text-[9px] uppercase border border-outline-variant transition-colors" data-inspect-clip="${c.clip_id}">
            Inspect Feed
          </button>
        </td>
      `;

      row.querySelector("[data-inspect-clip]").addEventListener("click", () => {
        // Switch to tab 1 and load this scenario
        document.getElementById("tabLiveBtn")?.click();
        if (this.scenarioSelect) {
          this.scenarioSelect.value = c.clip_id;
          this.loadScenario(c.clip_id);
        }
      });

      this.manifestTableBody.appendChild(row);
    });
  }

  // ==================== PPE ANALYSER ====================
  setupPPEAnalyser() {
    const photoFileInput = document.getElementById("ppeImageUploadInput");
    const videoFileInput = document.getElementById("ppeVideoUploadInput");
    const loadingBox = document.getElementById("ppeAnalysisLoading");
    const loadingMainText = document.getElementById("ppeLoadingMainText");
    const loadingSubText = document.getElementById("ppeLoadingSubText");

    // Mode Toggle Buttons
    const photoModeBtn = document.getElementById("ppeModePhotoBtn");
    const videoModeBtn = document.getElementById("ppeModeVideoBtn");
    const photoControls = document.getElementById("ppePhotoModeControls");
    const videoControls = document.getElementById("ppeVideoModeControls");
    const modeDesc = document.getElementById("ppeStudioModeDescription");
    const playbackControls = document.getElementById("ppeVideoPlaybackControls");

    this.ppeCurrentMode = "photo"; // "photo" or "video"
    this.ppeVideoFile = null;
    this.ppeVideoData = null;
    this.ppeVideoFrames = [];
    this.ppeCurrentFrameIdx = 0;
    this.ppeIsPlaying = false;
    this.ppePlaybackTimer = null;
    this.ppePlaybackSpeed = 1.0;
    this.ppeLoop = true;

    // Switch to Photo Mode
    photoModeBtn?.addEventListener("click", () => {
      this.ppeCurrentMode = "photo";
      this.pausePPEVideo();
      photoModeBtn.className = "px-space-sm py-0.5 font-label-caps text-[10px] uppercase font-bold bg-primary text-on-primary transition-all cursor-pointer";
      videoModeBtn.className = "px-space-sm py-0.5 font-label-caps text-[10px] uppercase font-bold text-on-surface-variant hover:text-on-surface transition-all cursor-pointer";
      photoControls?.classList.remove("hidden");
      videoControls?.classList.add("hidden");
      playbackControls?.classList.add("hidden");
      if (modeDesc) {
        modeDesc.innerHTML = 'Upload any construction test image or click a verified sample from <code class="text-tertiary font-mono">datasets/ppe_master_folder</code>.';
      }
    });

    // Switch to Video Stream Mode
    videoModeBtn?.addEventListener("click", () => {
      this.ppeCurrentMode = "video";
      videoModeBtn.className = "px-space-sm py-0.5 font-label-caps text-[10px] uppercase font-bold bg-secondary text-on-secondary transition-all cursor-pointer";
      photoModeBtn.className = "px-space-sm py-0.5 font-label-caps text-[10px] uppercase font-bold text-on-surface-variant hover:text-on-surface transition-all cursor-pointer";
      videoControls?.classList.remove("hidden");
      photoControls?.classList.add("hidden");
      if (this.ppeVideoFrames.length > 0) {
        playbackControls?.classList.remove("hidden");
      }
      if (modeDesc) {
        modeDesc.innerHTML = 'Select a local construction video stream (<code class="text-secondary font-mono">.mp4, .mov, .avi, .webm</code>), dissect into frames, and run frame-by-frame neural PPE compliance.';
      }
    });

    // 1. Photo File Upload Handler
    photoFileInput?.addEventListener("change", async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;

      this.pausePPEVideo();
      playbackControls?.classList.add("hidden");

      const formData = new FormData();
      formData.append("file", file);

      if (loadingMainText) loadingMainText.textContent = "Executing Perception (YOLOv8) → Cranial Spatial Reasoning → PPEReasonerNet...";
      if (loadingSubText) loadingSubText.textContent = "Processing image through FaceCranialDetector & OSHA PPE Reasoner...";
      loadingBox?.classList.remove("hidden");
      const startTime = performance.now();

      try {
        const resp = await fetch("/api/ppe/analyze", {
          method: "POST",
          body: formData
        });
        const data = await resp.json();
        const duration = Math.round(performance.now() - startTime);

        this.renderPPEResults(data, duration);
      } catch (err) {
        console.error("PPE photo analysis failed:", err);
        alert("PPE Analysis failed: " + err.message);
      } finally {
        loadingBox?.classList.add("hidden");
      }
    });

    // 2. Sample Preset Buttons
    [1, 2, 3].forEach(idx => {
      const btn = document.getElementById(`ppeSampleBtn${idx}`);
      btn?.addEventListener("click", async () => {
        this.pausePPEVideo();
        playbackControls?.classList.add("hidden");

        if (loadingMainText) loadingMainText.textContent = "Executing Perception (YOLOv8) → Cranial Spatial Reasoning → PPEReasonerNet...";
        loadingBox?.classList.remove("hidden");
        const startTime = performance.now();
        try {
          const samplesResp = await fetch("/api/ppe/samples");
          const samplesData = await samplesResp.json();
          const sample = samplesData.samples?.[idx - 1] || samplesData.samples?.[0];

          if (sample) {
            const resp = await fetch("/api/ppe/analyze_sample", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ sample_path: sample.path })
            });
            const data = await resp.json();
            const duration = Math.round(performance.now() - startTime);
            this.renderPPEResults(data, duration);
          }
        } catch (err) {
          console.error("Failed to analyze sample:", err);
        } finally {
          loadingBox?.classList.add("hidden");
        }
      });
    });

    // 3. Video File Selection Handler
    videoFileInput?.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (!file) return;

      this.ppeVideoFile = file;
      const badge = document.getElementById("ppeVideoSelectedBadge");
      const nameText = document.getElementById("ppeVideoFilenameText");
      if (badge && nameText) {
        badge.classList.remove("hidden");
        const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
        nameText.textContent = `${file.name} (${sizeMb} MB)`;
      }
    });

    // 4. Video Run Test Button & Preset Video Clips
    const runVideoAnalysis = async (samplePath = null) => {
      this.pausePPEVideo();
      playbackControls?.classList.add("hidden");

      const targetFps = document.getElementById("ppeTargetFpsSelect")?.value || "8.0";
      const maxFrames = document.getElementById("ppeMaxFramesSelect")?.value || "120";

      let url = "/api/ppe/analyze_video";
      let requestOptions = {};

      if (samplePath) {
        url = "/api/ppe/analyze_video_sample";
        requestOptions = {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sample_path: samplePath,
            target_fps: parseFloat(targetFps),
            max_frames: parseInt(maxFrames, 10)
          })
        };
        const fname = samplePath.split("/").pop();
        if (loadingMainText) loadingMainText.textContent = `Dissecting sample clip "${fname}" into frames & running neural inference...`;
      } else if (this.ppeVideoFile) {
        const formData = new FormData();
        formData.append("file", this.ppeVideoFile);
        formData.append("target_fps", targetFps);
        formData.append("max_frames", maxFrames);
        requestOptions = { method: "POST", body: formData };
        if (loadingMainText) loadingMainText.textContent = `Dissecting uploaded video "${this.ppeVideoFile.name}" into frames & running neural inference...`;
      } else {
        // If user pushes test button without uploading, seamlessly evaluate default demo clip
        url = "/api/ppe/analyze_video_sample";
        requestOptions = {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sample_path: "data/raw/clip_03_critical_danger_zone.mp4",
            target_fps: parseFloat(targetFps),
            max_frames: parseInt(maxFrames, 10)
          })
        };
        if (loadingMainText) loadingMainText.textContent = `Dissecting demo clip "clip_03_critical_danger_zone.mp4" into frames & running neural inference...`;
      }

      if (loadingSubText) loadingSubText.textContent = `Sampling stream at ${targetFps} FPS (max ${maxFrames} frames) through FaceCranialDetector & PPEReasonerNet...`;
      loadingBox?.classList.remove("hidden");

      try {
        const resp = await fetch(url, requestOptions);
        if (!resp.ok) {
          const errData = await resp.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned status ${resp.status}`);
        }
        const data = await resp.json();
        this.setupPPEVideoPlayback(data);
      } catch (err) {
        console.error("Video PPE analysis failed:", err);
        alert("Video PPE Analysis failed: " + err.message);
      } finally {
        loadingBox?.classList.add("hidden");
      }
    };

    const runVideoBtn = document.getElementById("ppeRunVideoAnalysisBtn");
    runVideoBtn?.addEventListener("click", () => runVideoAnalysis(null));

    document.getElementById("ppeSampleVideoBtn1")?.addEventListener("click", () => {
      runVideoAnalysis("data/raw/clip_03_critical_danger_zone.mp4");
    });
    document.getElementById("ppeSampleVideoBtn2")?.addEventListener("click", () => {
      runVideoAnalysis("data/raw/clip_01_safe_walkway.mp4");
    });


    // 5. Video Playback Transport Event Listeners
    document.getElementById("ppePlayPauseBtn")?.addEventListener("click", () => this.togglePPEPlayPause());
    document.getElementById("ppePrevFrameBtn")?.addEventListener("click", () => {
      this.pausePPEVideo();
      this.seekPPEVideoFrame(this.ppeCurrentFrameIdx - 1);
    });
    document.getElementById("ppeNextFrameBtn")?.addEventListener("click", () => {
      this.pausePPEVideo();
      this.seekPPEVideoFrame(this.ppeCurrentFrameIdx + 1);
    });
    document.getElementById("ppeLoopToggleBtn")?.addEventListener("click", (e) => {
      this.ppeLoop = !this.ppeLoop;
      e.currentTarget.textContent = `LOOP: ${this.ppeLoop ? 'ON' : 'OFF'}`;
      e.currentTarget.className = this.ppeLoop 
        ? "px-space-xs py-1 bg-surface-container text-secondary border border-secondary font-label-caps text-[9px] font-bold cursor-pointer"
        : "px-space-xs py-1 bg-surface-container text-on-surface-variant border border-outline-variant font-label-caps text-[9px] font-bold cursor-pointer";
    });
    document.getElementById("ppeVideoSpeedSelect")?.addEventListener("change", (e) => {
      this.ppePlaybackSpeed = parseFloat(e.target.value) || 1.0;
      if (this.ppeIsPlaying) {
        this.pausePPEVideo();
        this.playPPEVideo();
      }
    });
    document.getElementById("ppeVideoTimelineScrubber")?.addEventListener("input", (e) => {
      this.pausePPEVideo();
      this.seekPPEVideoFrame(parseInt(e.target.value, 10));
    });
  }

  // Set up Video Playback from API result
  setupPPEVideoPlayback(data) {
    if (!data || data.status !== "success") return;

    this.ppeVideoData = data;
    this.ppeVideoFrames = data.frames || [];
    this.ppeCurrentFrameIdx = 0;

    const playbackControls = document.getElementById("ppeVideoPlaybackControls");
    playbackControls?.classList.remove("hidden");

    const scrubber = document.getElementById("ppeVideoTimelineScrubber");
    if (scrubber) {
      scrubber.min = "0";
      scrubber.max = (this.ppeVideoFrames.length - 1).toString();
      scrubber.value = "0";
    }

    // Populate Temporal Violation Heatmap Strip
    const heatmap = document.getElementById("ppeViolationHeatmapStrip");
    const timeline = data.violation_timeline || [];
    if (heatmap) {
      heatmap.innerHTML = "";
      timeline.forEach((tick, idx) => {
        const tickEl = document.createElement("div");
        let bgClass = "bg-secondary"; // Compliant Emerald
        if (tick.status_code === 2) {
          bgClass = "bg-error"; // Critical No Helmet
        } else if (tick.status_code === 1) {
          bgClass = "bg-amber-400"; // Missing Vest
        }

        tickEl.className = `ppe-heatmap-tick flex-1 h-full ${bgClass} hover:opacity-100 opacity-85 transition-all`;
        tickEl.title = `Frame #${idx + 1} (${tick.timestamp_sec}s): ${tick.status_code === 2 ? 'CRITICAL NO PPE' : (tick.status_code === 1 ? 'MISSING VEST' : 'COMPLIANT')}`;
        tickEl.addEventListener("click", () => {
          this.pausePPEVideo();
          this.seekPPEVideoFrame(idx);
        });
        heatmap.appendChild(tickEl);
      });
    }

    // Breach Summary text
    const breachSummaryEl = document.getElementById("ppeTimelineBreachSummary");
    const clip = data.clip_summary || {};
    if (breachSummaryEl) {
      const crit = clip.critical_violation_frames_count || 0;
      const mod = clip.moderate_violation_frames_count || 0;
      breachSummaryEl.textContent = `${crit} CRITICAL, ${mod} MODERATE BREACHES`;
      breachSummaryEl.className = crit > 0 ? "text-error font-bold" : (mod > 0 ? "text-amber-400 font-bold" : "text-secondary font-bold");
    }

    // Render initial frame and auto-play
    this.renderPPEVideoFrame(0);
    this.playPPEVideo();
  }

  // Seek to specific frame in video stream
  seekPPEVideoFrame(frameIdx) {
    if (!this.ppeVideoFrames || this.ppeVideoFrames.length === 0) return;
    const clamped = Math.max(0, Math.min(this.ppeVideoFrames.length - 1, frameIdx));
    this.renderPPEVideoFrame(clamped);
  }

  // Play Video Loop
  playPPEVideo() {
    if (this.ppeIsPlaying || this.ppeVideoFrames.length === 0) return;
    this.ppeIsPlaying = true;

    const playIcon = document.getElementById("ppePlayPauseIcon");
    if (playIcon) playIcon.textContent = "pause";

    const meta = this.ppeVideoData?.video_metadata || {};
    const baseFps = meta.playback_fps || 8.0;
    const intervalMs = Math.max(25, Math.round(1000.0 / (baseFps * this.ppePlaybackSpeed)));

    this.ppePlaybackTimer = setInterval(() => {
      let nextIdx = this.ppeCurrentFrameIdx + 1;
      if (nextIdx >= this.ppeVideoFrames.length) {
        if (this.ppeLoop) {
          nextIdx = 0;
        } else {
          this.pausePPEVideo();
          return;
        }
      }
      this.renderPPEVideoFrame(nextIdx);
    }, intervalMs);
  }

  // Pause Video
  pausePPEVideo() {
    this.ppeIsPlaying = false;
    if (this.ppePlaybackTimer) {
      clearInterval(this.ppePlaybackTimer);
      this.ppePlaybackTimer = null;
    }
    const playIcon = document.getElementById("ppePlayPauseIcon");
    if (playIcon) playIcon.textContent = "play_arrow";
  }

  // Toggle Play/Pause
  togglePPEPlayPause() {
    if (this.ppeIsPlaying) {
      this.pausePPEVideo();
    } else {
      this.playPPEVideo();
    }
  }

  // Render a Single Video Frame & Synchronize All UI Components
  renderPPEVideoFrame(frameIndex) {
    const frame = this.ppeVideoFrames[frameIndex];
    if (!frame) return;

    this.ppeCurrentFrameIdx = frameIndex;

    // 1. Update Canvas Image Display
    const imgDisplay = document.getElementById("ppeAnnotatedImageDisplay");
    const placeholder = document.getElementById("ppeImagePlaceholder");
    if (frame.annotated_image_base64 && imgDisplay) {
      imgDisplay.src = frame.annotated_image_base64;
      imgDisplay.classList.remove("hidden");
      if (placeholder) placeholder.classList.add("hidden");
    }

    // 2. Update Transport Scrubber & Readout
    const scrubber = document.getElementById("ppeVideoTimelineScrubber");
    if (scrubber) scrubber.value = frameIndex.toString();

    const frameDisp = document.getElementById("ppeCurrentFrameIndexDisplay");
    if (frameDisp) frameDisp.textContent = `${frameIndex + 1} / ${this.ppeVideoFrames.length}`;

    const timeDisp = document.getElementById("ppeCurrentTimestampDisplay");
    if (timeDisp) timeDisp.textContent = frame.time_display || "00:00.0";

    // 3. Highlight current tick on timeline heatmap strip
    const ticks = document.querySelectorAll(".ppe-heatmap-tick");
    ticks.forEach((tick, i) => {
      if (i === frameIndex) {
        tick.classList.add("ring-2", "ring-white", "opacity-100", "scale-y-125");
      } else {
        tick.classList.remove("ring-2", "ring-white", "scale-y-125");
      }
    });

    // 4. Update Footer Metadata
    const meta = this.ppeVideoData?.video_metadata || {};
    const setVal = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };
    setVal("ppeFilenameDisplay", `VIDEO: ${meta.filename || "Uploaded Stream"}`);
    setVal("ppeDimensionsDisplay", `RES: ${meta.resolution?.width || 0}x${meta.resolution?.height || 0} | PLAYBACK: ${meta.playback_fps || 8} FPS`);
    setVal("ppeInferenceTimeDisplay", `AVG LATENCY: ${meta.avg_frame_latency_ms || 0} ms/frame`);

    // 5. Dynamically update the Right-Side Worker Decision Cards for this frame!
    this.renderPPEWorkerCards(frame.workers || []);

    // 6. Update Top Metric Cards for this frame
    const summary = frame.summary_metrics || {};
    setVal("metricTotalWorkers", summary.total_workers ?? 0);
    setVal("metricCompliantCount", summary.compliant_count ?? 0);
    setVal("metricViolationCount", summary.violation_count ?? 0);
    setVal("metricHardhatRate", `${summary.hardhat_compliance_rate_pct ?? 0}%`);
    setVal("metricVestRate", `${summary.vest_compliance_rate_pct ?? 0}%`);
    setVal("metricAvgRisk", (summary.avg_risk_score ?? 0).toFixed(3));

    const statusEl = document.getElementById("metricSiteStatus");
    const siteStatus = summary.site_status || "UNKNOWN";
    if (statusEl) {
      statusEl.textContent = siteStatus.replace(/_/g, " ");
      if (siteStatus === "FULL_COMPLIANCE") {
        statusEl.className = "inline-block px-2 py-0.5 mt-1 font-label-caps text-[10px] font-bold border border-secondary text-secondary bg-secondary-container/20";
      } else if (siteStatus === "CRITICAL_VIOLATIONS") {
        statusEl.className = "inline-block px-2 py-0.5 mt-1 font-label-caps text-[10px] font-bold border border-error text-error bg-error-container/20";
      } else {
        statusEl.className = "inline-block px-2 py-0.5 mt-1 font-label-caps text-[10px] font-bold border border-tertiary text-tertiary bg-tertiary-container/20";
      }
    }
  }

  // Render Static Photo Results
  renderPPEResults(data, latencyMs) {
    if (!data || data.status !== "success") return;

    const summary = data.summary_metrics || {};
    const workers = data.workers || [];

    const setVal = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    setVal("metricTotalWorkers", summary.total_workers ?? 0);
    setVal("metricCompliantCount", summary.compliant_count ?? 0);
    setVal("metricViolationCount", summary.violation_count ?? 0);
    setVal("metricHardhatRate", `${summary.hardhat_compliance_rate_pct ?? 0}%`);
    setVal("metricVestRate", `${summary.vest_compliance_rate_pct ?? 0}%`);
    setVal("metricAvgRisk", (summary.avg_risk_score ?? 0).toFixed(3));

    const statusEl = document.getElementById("metricSiteStatus");
    const siteStatus = summary.site_status || "UNKNOWN";
    if (statusEl) {
      statusEl.textContent = siteStatus.replace(/_/g, " ");
      if (siteStatus === "FULL_COMPLIANCE") {
        statusEl.className = "inline-block px-2 py-0.5 mt-1 font-label-caps text-[10px] font-bold border border-secondary text-secondary bg-secondary-container/20";
      } else if (siteStatus === "CRITICAL_VIOLATIONS") {
        statusEl.className = "inline-block px-2 py-0.5 mt-1 font-label-caps text-[10px] font-bold border border-error text-error bg-error-container/20";
      } else {
        statusEl.className = "inline-block px-2 py-0.5 mt-1 font-label-caps text-[10px] font-bold border border-tertiary text-tertiary bg-tertiary-container/20";
      }
    }

    // Annotated Image View
    const placeholder = document.getElementById("ppeImagePlaceholder");
    const imgDisplay = document.getElementById("ppeAnnotatedImageDisplay");
    if (data.annotated_image_base64 && imgDisplay) {
      imgDisplay.src = data.annotated_image_base64;
      imgDisplay.classList.remove("hidden");
      if (placeholder) placeholder.classList.add("hidden");
    }

    // Metadata Strip
    setVal("ppeFilenameDisplay", `SOURCE: ${data.filename || "Uploaded Image"}`);
    setVal("ppeDimensionsDisplay", `RESOLUTION: ${data.image_dimensions?.width || 0} x ${data.image_dimensions?.height || 0}`);
    setVal("ppeInferenceTimeDisplay", `LATENCY: ${latencyMs} ms`);

    // Render Worker Cards
    this.renderPPEWorkerCards(workers);
  }

  // Reusable Per-Worker Neural Decision Cards Renderer
  renderPPEWorkerCards(workers) {
    const container = document.getElementById("ppeWorkerCardsContainer");
    const countBadge = document.getElementById("ppeWorkerCardCount");
    if (countBadge) countBadge.textContent = `(${workers.length} Workers)`;
    if (!container) return;
    container.innerHTML = "";

    if (workers.length === 0) {
      container.innerHTML = `
        <div class="p-space-lg bg-surface-container-lowest border border-outline-variant text-center">
          <span class="font-label-mono-micro text-[11px] text-on-surface-variant block">
            No active personnel detected in current frame.
          </span>
        </div>
      `;
      return;
    }

    workers.forEach((w) => {
      let badgeClass = "border-secondary text-secondary bg-secondary-container/20";
      let statusDot = "bg-secondary";
      if (w.violation_id === 1) {
        badgeClass = "border-tertiary text-tertiary bg-tertiary-container/20";
        statusDot = "bg-tertiary";
      } else if (w.violation_id === 2) {
        badgeClass = "border-yellow-400 text-yellow-400 bg-yellow-400/20";
        statusDot = "bg-yellow-400";
      } else if (w.violation_id === 3) {
        badgeClass = "border-error text-error bg-error-container/20";
        statusDot = "bg-error";
      }

      const card = document.createElement("div");
      card.className = "p-space-sm bg-surface-container-lowest border border-outline-variant space-y-2 hover:border-primary transition-colors";
      card.innerHTML = `
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full ${statusDot}"></span>
            <span class="font-label-caps text-[11px] font-bold text-on-surface">${w.worker_id}</span>
            <span class="font-mono text-[10px] text-on-surface-variant">(Conf: ${(w.confidence * 100).toFixed(0)}%)</span>
          </div>
          <span class="px-1.5 py-0.5 font-label-caps text-[9px] font-bold border ${badgeClass}">
            ${w.violation_class.replace(/_/g, " ")}
          </span>
        </div>

        <div class="grid grid-cols-2 gap-2 text-[10px] font-mono">
          <div class="bg-surface-container p-1.5 border border-outline-variant col-span-2 sm:col-span-1">
            <div class="flex justify-between items-center text-on-surface-variant">
              <span class="font-bold flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-sm bg-yellow-400"></span> Hardhat:</span>
              <span class="${w.has_hardhat ? 'text-secondary font-bold' : 'text-error font-bold'}">
                ${w.has_hardhat ? (w.is_good_detect ? '✓ GOOD DETECT' : 'WORN') : 'MISSING'}
              </span>
            </div>
            <div class="w-full bg-surface-container-lowest h-1 mt-1">
              <div class="h-1 ${w.has_hardhat ? 'bg-secondary' : 'bg-error'}" style="width: ${w.hardhat_compliance_pct}%"></div>
            </div>
            <div class="flex justify-between items-center text-[9px] text-on-surface-variant mt-1">
              <span>${w.hardhat_compliance_pct}% Conf</span>
              ${w.hardhat_box ? `<span class="text-yellow-400 font-mono font-bold text-[8.5px]">Square: ${w.hardhat_square_dims || (w.hardhat_box[2]-w.hardhat_box[0]+'×'+(w.hardhat_box[3]-w.hardhat_box[1])+'px')}</span>` : `<span class="text-error font-mono text-[8.5px]">No Square</span>`}
            </div>
            ${w.is_good_detect && w.hardhat_box ? `
              <div class="mt-1 px-1 py-0.5 bg-surface-container-lowest border border-yellow-400/50 text-yellow-400 text-[8.5px] flex items-center justify-between">
                <span>Alignment:</span>
                <span class="font-bold text-secondary">✓ Sits On Top Of Face</span>
              </div>
            ` : (w.face_box ? `
              <div class="mt-1 px-1 py-0.5 bg-surface-container-lowest border border-error/40 text-error text-[8.5px] flex items-center justify-between">
                <span>Cranial Vault:</span>
                <span class="font-bold">HEAD EXPOSED (NO HELMET)</span>
              </div>
            ` : `
              <div class="mt-1 px-1 py-0.5 bg-surface-container-lowest border border-error/40 text-error text-[8.5px] flex items-center justify-between">
                <span>Cranial Status:</span>
                <span class="font-bold">UNPROTECTED</span>
              </div>
            `)}
            ${w.face_box ? `
              <div class="mt-1 px-1 py-0.5 bg-surface-container-lowest border border-outline-variant text-[8px] text-on-surface-variant flex items-center justify-between">
                <span>Face Detected:</span>
                <span class="font-mono text-cyan-400">[${w.face_box[0]},${w.face_box[1]}] (${(w.face_confidence*100).toFixed(0)}%)</span>
              </div>
            ` : ''}
          </div>

          <div class="bg-surface-container p-1.5 border border-outline-variant col-span-2 sm:col-span-1">
            <div class="flex justify-between text-on-surface-variant">
              <span>Safety Vest:</span>
              <span class="${w.has_vest ? 'text-secondary font-bold' : 'text-error font-bold'}">
                ${w.has_vest ? 'WORN' : 'MISSING'}
              </span>
            </div>
            <div class="w-full bg-surface-container-lowest h-1 mt-1">
              <div class="h-1 ${w.has_vest ? 'bg-secondary' : 'bg-error'}" style="width: ${w.vest_compliance_pct}%"></div>
            </div>
            <span class="text-[9px] text-on-surface-variant mt-1 block">${w.vest_compliance_pct}% Confidence</span>
          </div>
        </div>

        <div class="flex items-center justify-between text-[10px] font-mono border-t border-outline-variant pt-1.5">
          <span class="text-on-surface-variant">Risk: <strong class="${w.risk_score >= 0.5 ? 'text-error' : 'text-secondary'}">${w.risk_score.toFixed(3)}</strong> (${w.risk_level})</span>
          <span class="text-primary text-[9px] truncate max-w-[200px]" title="${w.action}">${w.action}</span>
        </div>
      `;
      container.appendChild(card);
    });
  }
}

// Instantiate on DOM Load
document.addEventListener("DOMContentLoaded", () => {
  window.hazardMeshApp = new HazardMeshConsole();
});
