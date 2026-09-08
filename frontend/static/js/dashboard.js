/**
 * NeuroGuard Clinic – Dashboard Controller (Real Data)
 * Camera frame capture, MediaPipe overlay, MAX30102 sensor polling,
 * AI-personalized questions, real sensor status management.
 */

// ============================================================
// State
// ============================================================
let testRunning = false;
let testPaused = false;
let timerInterval = null;
let pollInterval = null;
let frameInterval = null;
let elapsedSeconds = 0;
let cameraStream = null;
let currentSessionId = null;
let currentPatientId = document.getElementById('patientId')?.textContent || 'PAT-???';

// Question state
let answeredCount = 0;
let currentQuestionIndex = 0;
let allQuestions = [...QUESTIONS]; // from Jinja template
let totalQuestions = TOTAL_QUESTIONS;
let waitingForAI = false;

// Sensor status tracking
let sensorState = {
    camera: false,
    max30102: false,
    finger: false,
};

// Chart instances
let hrvChart, fusionChart, emgChart;
const MAX_POINTS = 40;
const hrvBpm = [];
const fusionScores = [];
const emgValues = [];
const timeLabels = [];

// ============================================================
// Init on DOM Ready
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initCamera();
    updateSensorStatusFromServer();
    updateFallbackNotice();
    updateSerialMonitor();
    // Periodically refresh sensor status even before test starts
    setInterval(updateSensorStatusFromServer, 3000);
    setInterval(updateSerialMonitor, 1000);
});

// ============================================================
// Camera Init
// ============================================================
async function initCamera() {
    const video = document.getElementById('cameraFeed');
    const offlineMsg = document.getElementById('cameraOfflineMsg');
    const statusBadge = document.getElementById('cameraStatusBadge');

    if (!window.isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
        console.error('Camera requires a secure context (HTTPS or localhost)');
        offlineMsg.style.display = 'flex';
        const reason = offlineMsg.querySelector('small');
        if (reason) reason.textContent = 'Bypass: Use localhost instead of IP for camera access.';
        setSensorOnline('camera', false, 'Requires HTTPS or localhost');
        return;
    }

    try {
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };

        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ ...constraints, video: { ...constraints.video, facingMode: 'user' } });
        } catch (e) {
            console.warn('facingMode:user failed, trying generic video constraints', e);
            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        }

        video.srcObject = cameraStream;
        video.style.transform = 'scaleX(-1)';
        video.style.display = 'block';

        video.onloadedmetadata = () => {
            video.play();
            // Start heartbeat analyzer (1 frame every 3s) to keep backend status 'online'
            setInterval(() => {
                if (!testRunning && cameraStream) captureAndAnalyzeFrame();
            }, 3000);
        };

        offlineMsg.style.display = 'none';
        statusBadge.className = 'badge bg-success-subtle text-success';
        statusBadge.innerHTML = '<i class="bi bi-circle-fill blink-dot me-1"></i>LIVE';
        setSensorOnline('camera', true);

    } catch (err) {
        console.error('Camera init error:', err);
        video.style.display = 'none';
        offlineMsg.style.display = 'flex';
        statusBadge.className = 'badge bg-danger-subtle text-danger';
        statusBadge.innerHTML = '<i class="bi bi-camera-video-off me-1"></i>OFFLINE';
        
        let errorMsg = err.message;
        if (err.name === 'NotAllowedError') errorMsg = 'Permission denied. Please allow camera access.';
        else if (err.name === 'NotFoundError') errorMsg = 'No camera found on this device.';
        
        setSensorOnline('camera', false, errorMsg);

        const offlineReason = document.querySelector('#cameraOfflineMsg small');
        if (offlineReason) offlineReason.textContent = errorMsg;
    }
}

// ============================================================
// Camera Frame Capture & Overlay
// ============================================================
const captureCanvas = document.createElement('canvas');
captureCanvas.width = 320;
captureCanvas.height = 240;
const captureCtx = captureCanvas.getContext('2d');

async function captureAndAnalyzeFrame() {
    const video = document.getElementById('cameraFeed');
    if (!cameraStream || !video.videoWidth) return;

    // Draw mirrored frame to capture canvas
    captureCtx.save();
    captureCtx.translate(captureCanvas.width, 0);
    captureCtx.scale(-1, 1);
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    captureCtx.restore();

    const frameData = captureCanvas.toDataURL('image/jpeg', 0.7);

    try {
        const res = await fetch('/api/camera_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame: frameData }),
        });
        const result = await res.json();
        drawOverlay(result);
    } catch (err) {
        // Silently fail – don't spam errors on frame drops
    }
}

function drawOverlay(analysisResult) {
    const overlay = document.getElementById('cameraOverlay');
    if (!overlay) return;
    const ctx = overlay.getContext('2d');
    const w = overlay.width;
    const h = overlay.height;

    ctx.clearRect(0, 0, w, h);

    if (!analysisResult || !analysisResult.face_detected) return;

    // Draw face landmarks as green dots (mirrored to match video)
    const landmarks = analysisResult.landmarks || [];
    ctx.fillStyle = 'rgba(0, 229, 255, 0.8)';
    for (const pt of landmarks) {
        // Mirror x because video is mirrored
        const x = (1 - pt[0]) * w;
        const y = pt[1] * h;
        ctx.beginPath();
        ctx.arc(x, y, 1.2, 0, Math.PI * 2);
        ctx.fill();
    }

    // Pose skeleton (upper body)
    const pose = analysisResult.pose_landmarks || [];
    if (pose.length > 0) {
        const CONNECTIONS = [[11,12],[11,13],[12,14],[13,15],[14,16],[11,23],[12,24]];
        ctx.strokeStyle = 'rgba(255, 200, 0, 0.6)';
        ctx.lineWidth = 2;
        for (const [a, b] of CONNECTIONS) {
            if (pose[a] && pose[b]) {
                ctx.beginPath();
                ctx.moveTo((1 - pose[a][0]) * w, pose[a][1] * h);
                ctx.lineTo((1 - pose[b][0]) * w, pose[b][1] * h);
                ctx.stroke();
            }
        }
    }

    // Status text overlay
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillStyle = 'rgba(0, 229, 255, 0.9)';
    ctx.fillText(`Blinks: ${analysisResult.blink_rate || 0}/min`, 8, 18);
    const expr = analysisResult.facial_expression || {};
    const dominant = Object.entries(expr).sort((a, b) => b[1] - a[1])[0];
    if (dominant) {
        ctx.fillText(`Expr: ${dominant[0]} (${dominant[1].toFixed(0)}%)`, 8, 34);
    }
    ctx.fillText(`Posture: ${analysisResult.posture_score || 0}%`, 8, 50);
}

// Sync overlay canvas size with video
function syncOverlaySize() {
    const video = document.getElementById('cameraFeed');
    const overlay = document.getElementById('cameraOverlay');
    if (video && overlay && video.videoWidth) {
        overlay.width = video.offsetWidth;
        overlay.height = video.offsetHeight;
    }
}
setInterval(syncOverlaySize, 2000);

// ============================================================
// Sensor Status Management
// ============================================================
function setSensorOnline(sensor, online, errorMsg) {
    sensorState[sensor] = online;

    // Update header sensor badge
    const badge = document.getElementById(`sensor-${sensor}`);
    if (badge) {
        const dot = badge.querySelector('.sensor-dot');
        if (dot) {
            dot.className = `sensor-dot ${online ? 'online' : 'offline'}`;
            badge.title = online ? 'Online' : (errorMsg || 'Offline');
        }
    }

    // Update pipeline panel
    const pipelineDot = document.querySelector(`#pipeline-${sensor} .pipeline-dot`);
    const pipelineBadge = document.getElementById(`pipeline-${sensor}-badge`);
    if (pipelineDot) {
        pipelineDot.className = `pipeline-dot ${online ? 'online' : 'offline'}`;
    }
    if (pipelineBadge) {
        if (online) {
            pipelineBadge.className = 'pipeline-status badge bg-success-subtle text-success';
            pipelineBadge.textContent = 'ONLINE';
        } else {
            pipelineBadge.className = 'pipeline-status badge bg-danger-subtle text-danger';
            pipelineBadge.textContent = 'OFFLINE';
        }
    }

    updateFallbackNotice();
}

function updateFallbackNotice() {
    const notice = document.getElementById('fallbackNotice');
    const msg = document.getElementById('fallbackMessage');
    if (!notice || !msg) return;

    const cameraOn = sensorState.camera;
    const max30102On = sensorState.max30102;

    if (cameraOn && max30102On) {
        notice.className = 'fallback-notice fallback-success';
        msg.innerHTML = '<strong>Full Multimodal Mode</strong> — All sensors active. Maximum accuracy.';
    } else if (cameraOn && !max30102On) {
        notice.className = 'fallback-notice fallback-warning';
        msg.innerHTML = '<strong>Camera-Only Mode</strong> — MAX30102 offline. Using expression + blink + posture + questionnaire.';
    } else if (!cameraOn && max30102On) {
        notice.className = 'fallback-notice fallback-warning';
        msg.innerHTML = '<strong>Sensor-Only Mode</strong> — Camera offline. Using HR + SpO₂ + HRV + questionnaire.';
    } else {
        notice.className = 'fallback-notice fallback-danger';
        msg.innerHTML = '<strong>Questions-Only Mode</strong> — No sensors connected. Results based on PHQ-9 answers only.';
    }
}

async function updateSensorStatusFromServer() {
    try {
        const res = await fetch('/api/sensor_data');
        const data = await res.json();
        const sensors = data.sensors || {};

        // Camera: use local stream status as truth
        if (cameraStream) {
            setSensorOnline('camera', true);
        } else if (sensors.camera) {
            setSensorOnline('camera', sensors.camera.online, sensors.camera.error);
        }

        // MAX30102: check ESP32 connection + finger detection separately
        const esp32Connected = data.esp32_connected || false;
        const fingerDetected = data.finger_detected || false;
        
        sensorState.max30102 = esp32Connected;
        sensorState.finger = fingerDetected;

        // Update MAX30102 panel UI
        updateMAX30102Panel(esp32Connected, fingerDetected, data.sensor_error);

        // Update header badges
        setSensorOnline('hrv', esp32Connected && fingerDetected);
        setSensorOnline('spo2', esp32Connected && fingerDetected);
        
        // Update pipeline for max30102
        const pipelineDot = document.querySelector('#pipeline-max30102 .pipeline-dot');
        const pipelineBadge = document.getElementById('pipeline-max30102-badge');
        if (pipelineDot) {
            pipelineDot.className = `pipeline-dot ${esp32Connected ? 'online' : 'offline'}`;
        }
        if (pipelineBadge) {
            if (esp32Connected && fingerDetected) {
                pipelineBadge.className = 'pipeline-status badge bg-success-subtle text-success';
                pipelineBadge.textContent = 'READING';
            } else if (esp32Connected) {
                pipelineBadge.className = 'pipeline-status badge bg-warning-subtle text-warning';
                pipelineBadge.textContent = 'NO FINGER';
            } else {
                pipelineBadge.className = 'pipeline-status badge bg-danger-subtle text-danger';
                pipelineBadge.textContent = 'OFFLINE';
            }
        }

        updateFallbackNotice();
    } catch (err) {
        // Ignore pre-start errors
    }
}

function updateMAX30102Panel(connected, fingerDetected, errorMsg) {
    const offlineOverlay = document.getElementById('max30102Offline');
    const liveData = document.getElementById('max30102Live');
    const statusBadge = document.getElementById('max30102Status');
    const errorEl = document.getElementById('max30102Error');
    const fingerStatus = document.getElementById('fingerStatus');

    if (!connected) {
        // ESP32 not connected
        if (offlineOverlay) offlineOverlay.style.display = 'flex';
        if (liveData) liveData.style.display = 'none';
        if (statusBadge) {
            statusBadge.className = 'badge bg-danger-subtle text-danger';
            statusBadge.innerHTML = '<i class="bi bi-usb-plug me-1"></i>NOT CONNECTED';
        }
        if (errorEl) errorEl.textContent = errorMsg || 'Connect ESP32 via USB to enable heart rate & SpO₂ monitoring';
    } else if (!fingerDetected) {
        // Connected but no finger
        if (offlineOverlay) offlineOverlay.style.display = 'none';
        if (liveData) liveData.style.display = 'block';
        if (statusBadge) {
            statusBadge.className = 'badge bg-warning-subtle text-warning';
            statusBadge.innerHTML = '<i class="bi bi-hand-index me-1"></i>NO FINGER';
        }
        if (fingerStatus) {
            fingerStatus.innerHTML = '<small class="text-warning"><i class="bi bi-hand-index me-1"></i>Place your finger firmly on the MAX30102 sensor</small>';
        }
    } else {
        // Fully online and reading
        if (offlineOverlay) offlineOverlay.style.display = 'none';
        if (liveData) liveData.style.display = 'block';
        if (statusBadge) {
            statusBadge.className = 'badge bg-success-subtle text-success';
            statusBadge.innerHTML = '<i class="bi bi-circle-fill blink-dot me-1"></i>READING';
        }
        if (fingerStatus) {
            fingerStatus.innerHTML = '<small class="text-success"><i class="bi bi-check-circle me-1"></i>Finger detected — reading vitals</small>';
        }
    }
}

function updateSensorDot(key, online, errorMsg) {
    const badge = document.getElementById(`sensor-${key}`);
    if (!badge) return;
    const dot = badge.querySelector('.sensor-dot');
    if (dot) {
        dot.className = `sensor-dot ${online ? 'online' : 'offline'}`;
        badge.title = online ? 'Online' : (errorMsg || 'Offline');
    }
}

// ============================================================
// Charts (Chart.js)
// ============================================================
function initCharts() {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        plugins: { legend: { display: false }, tooltip: { enabled: true } },
        scales: {
            x: { display: false, grid: { display: false } },
            y: {
                grid: { color: 'rgba(255,255,255,0.04)' },
                ticks: { color: 'rgba(255,255,255,0.3)', font: { size: 10 } }
            }
        },
        elements: { point: { radius: 0 }, line: { tension: 0.4 } }
    };

    const hrvCtx = document.getElementById('hrvChart').getContext('2d');
    hrvChart = new Chart(hrvCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'BPM',
                data: [],
                borderColor: '#f44336',
                backgroundColor: 'rgba(244,67,54,0.08)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: commonOptions
    });

    // EMG Chart
    const emgCtx = document.getElementById('emgChart');
    if (emgCtx) {
        emgChart = new Chart(emgCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'EMG',
                    data: [],
                    borderColor: '#ff9800',
                    backgroundColor: 'rgba(255,152,0,0.08)',
                    borderWidth: 2,
                    fill: true
                }]
            },
            options: commonOptions
        });
    }

    const fusionCtx = document.getElementById('fusionChart').getContext('2d');
    fusionChart = new Chart(fusionCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Fusion Score',
                data: [],
                borderColor: '#00e5ff',
                backgroundColor: 'rgba(0,229,255,0.07)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: { ...commonOptions.scales.y, min: 0, max: 27 }
            }
        }
    });
}

function pushDataPoint(data) {
    const label = data.timestamp || '';
    if (timeLabels.length >= MAX_POINTS) {
        timeLabels.shift(); hrvBpm.shift(); fusionScores.shift(); emgValues.shift();
    }
    timeLabels.push(label);

    const hrv = data.hrv;
    hrvBpm.push(hrv ? hrv.bpm : null);
    fusionScores.push(data.fusion_score || 0);

    // EMG
    const emg = data.emg;
    emgValues.push(emg ? emg.raw : null);

    hrvChart.data.labels = [...timeLabels];
    hrvChart.data.datasets[0].data = [...hrvBpm];
    hrvChart.update('none');

    fusionChart.data.labels = [...timeLabels];
    fusionChart.data.datasets[0].data = [...fusionScores];
    fusionChart.update('none');

    if (emgChart) {
        emgChart.data.labels = [...timeLabels];
        emgChart.data.datasets[0].data = [...emgValues];
        emgChart.update('none');
    }
}

// ============================================================
// Sensor Polling
// ============================================================
function startPolling() {
    pollInterval = setInterval(async () => {
        if (testPaused) return;
        try {
            const res = await fetch('/api/sensor_data');
            const data = await res.json();
            updateDashboard(data);
        } catch (err) {
            console.error('Polling error:', err);
        }
    }, 700);
}

function stopPolling() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

// ============================================================
// Frame Capture Loop (300ms for faster responsiveness)
// ============================================================
function startFrameCapture() {
    if (cameraStream) {
        frameInterval = setInterval(captureAndAnalyzeFrame, 300);
    }
}

function stopFrameCapture() {
    if (frameInterval) { clearInterval(frameInterval); frameInterval = null; }
}

// ============================================================
// Dashboard Update
// ============================================================
function updateDashboard(data) {
    pushDataPoint(data);

    const cam = data.camera || {};

    // Camera metrics
    const blinkEl = document.getElementById('blinkRate');
    if (blinkEl) blinkEl.textContent = cam.face_detected ? cam.blink_rate : '0';

    const expr = cam.facial_expression || {};
    const dominant = cam.face_detected && Object.keys(expr).length
        ? Object.entries(expr).sort((a, b) => b[1] - a[1])[0]
        : null;
    const faceEl = document.getElementById('facialExpression');
    if (faceEl) faceEl.textContent = dominant ? `${dominant[0].charAt(0).toUpperCase() + dominant[0].slice(1)}` : '--';

    const postureEl = document.getElementById('postureScore');
    if (postureEl) postureEl.textContent = cam.face_detected ? `${cam.posture_score}%` : '0%';

    // Expression bars
    ['neutral', 'sad', 'happy', 'anxious'].forEach(k => {
        const bar = document.getElementById(`expr${k.charAt(0).toUpperCase() + k.slice(1)}`);
        const pct = document.getElementById(`expr${k.charAt(0).toUpperCase() + k.slice(1)}Pct`);
        const val = cam.face_detected ? (expr[k] || 0) : 0;
        if (bar) bar.style.width = val + '%';
        if (pct) pct.textContent = val.toFixed(0) + '%';
    });

    // MAX30102 data
    const hrv = data.hrv;
    const bpmEl = document.getElementById('bpmValue');
    const rmssdEl = document.getElementById('rmssdValue');
    const spo2El = document.getElementById('spo2Value');
    if (bpmEl) bpmEl.textContent = (hrv && hrv.bpm !== undefined && hrv.bpm !== null) ? Math.round(hrv.bpm) : '--';
    if (rmssdEl) rmssdEl.textContent = (hrv && hrv.rmssd !== undefined && hrv.rmssd !== null) ? hrv.rmssd.toFixed(1) : '--';
    if (spo2El) spo2El.textContent = (data.spo2 !== undefined && data.spo2 !== null && data.spo2 > 0) ? data.spo2 + '%' : '--';

    // EMG data
    const emg = data.emg;
    const esp32Online = data.esp32_connected || false;
    updateEMGPanel(esp32Online, emg);

    // Update MAX30102 panel state
    const fingerOn = !!data.finger_detected;
    updateMAX30102Panel(esp32Online, fingerOn, data.sensor_error);

    // Update EMG pipeline
    const emgPipelineDot = document.querySelector('#pipeline-emg .pipeline-dot');
    const emgPipelineBadge = document.getElementById('pipeline-emg-badge');
    if (emgPipelineDot) emgPipelineDot.className = `pipeline-dot ${esp32Online ? 'online' : 'offline'}`;
    if (emgPipelineBadge) {
        if (esp32Online && emg && emg.raw > 0) {
            emgPipelineBadge.className = 'pipeline-status badge bg-success-subtle text-success';
            emgPipelineBadge.textContent = 'READING';
        } else if (esp32Online) {
            emgPipelineBadge.className = 'pipeline-status badge bg-warning-subtle text-warning';
            emgPipelineBadge.textContent = 'IDLE';
        } else {
            emgPipelineBadge.className = 'pipeline-status badge bg-danger-subtle text-danger';
            emgPipelineBadge.textContent = 'OFFLINE';
        }
    }

    // Fusion score
    const score = Math.round(data.fusion_score || 0);
    const scoreEl = document.querySelector('.score-value');
    if (scoreEl) {
        scoreEl.textContent = score;
        if (score <= 4) { scoreEl.style.color = '#4caf50'; updateSeverityBadge('Normal', 'success'); }
        else if (score <= 9) { scoreEl.style.color = '#ffeb3b'; updateSeverityBadge('Mild', 'warning'); }
        else if (score <= 14) { scoreEl.style.color = '#ff9800'; updateSeverityBadge('Moderate', 'orange'); }
        else if (score <= 19) { scoreEl.style.color = '#ff5722'; updateSeverityBadge('Mod. Severe', 'danger'); }
        else { scoreEl.style.color = '#f44336'; updateSeverityBadge('Severe', 'danger'); }
    }

    // Confidence
    const conf = data.confidence || 0;
    const confEl = document.getElementById('confidenceText');
    if (confEl) confEl.textContent = `Confidence: ${conf.toFixed(1)}%`;
    const liveConf = document.querySelector('#liveConfidence strong');
    if (liveConf) liveConf.textContent = `${conf.toFixed(1)}%`;

    // Update hover AI Analysis tab
    const hoverSev = document.getElementById('hoverAiSeverity');
    const hoverConf = document.getElementById('hoverAiConfidence');
    if (hoverSev) {
        if (score <= 4) hoverSev.textContent = 'Normal';
        else if (score <= 9) hoverSev.textContent = 'Mild';
        else if (score <= 14) hoverSev.textContent = 'Moderate';
        else if (score <= 19) hoverSev.textContent = 'Mod. Severe';
        else hoverSev.textContent = 'Severe';
    }
    if (hoverConf) hoverConf.textContent = `${conf.toFixed(1)}%`;

    // Update sensor dots from server response
    const sensors = data.sensors || {};
    for (const [key, info] of Object.entries(sensors)) {
        if (key === 'camera' && cameraStream) {
            updateSensorDot(key, true);
        } else {
            updateSensorDot(key, info.online, info.error);
        }
    }
}

function updateEMGPanel(connected, emgData) {
    const offlineOverlay = document.getElementById('emgOffline');
    const liveData = document.getElementById('emgLive');
    const statusBadge = document.getElementById('emgStatus');

    if (!connected) {
        if (offlineOverlay) offlineOverlay.style.display = 'flex';
        if (liveData) liveData.style.display = 'none';
        if (statusBadge) {
            statusBadge.className = 'badge bg-danger-subtle text-danger';
            statusBadge.innerHTML = '<i class="bi bi-usb-plug me-1"></i>NOT CONNECTED';
        }
        return;
    }

    if (offlineOverlay) offlineOverlay.style.display = 'none';
    if (liveData) liveData.style.display = 'block';
    if (statusBadge) {
        statusBadge.className = 'badge bg-success-subtle text-success';
        statusBadge.innerHTML = '<i class="bi bi-circle-fill blink-dot me-1"></i>READING';
    }

    if (!emgData) return;

    const rawEl = document.getElementById('emgRawValue');
    const avgEl = document.getElementById('emgAvgValue');
    const tensionPctEl = document.getElementById('emgTensionPct');
    const tensionBar = document.getElementById('emgTensionBar');
    const muscleIndicator = document.getElementById('muscleIndicator');
    const muscleIcon = document.getElementById('muscleIcon');
    const muscleLabel = document.getElementById('muscleLabel');

    if (rawEl) rawEl.textContent = emgData.raw || 0;
    if (avgEl) avgEl.textContent = emgData.avg || 0;
    if (tensionPctEl) tensionPctEl.textContent = (emgData.tension_pct || 0) + '%';
    if (tensionBar) tensionBar.style.width = Math.min(emgData.tension_pct || 0, 100) + '%';

    if (muscleIndicator && muscleIcon && muscleLabel) {
        if (emgData.muscle_active) {
            muscleIndicator.className = 'emg-muscle-indicator active';
            muscleLabel.textContent = 'Active';
        } else {
            muscleIndicator.className = 'emg-muscle-indicator';
            muscleLabel.textContent = 'Relaxed';
        }
    }
}

function updateSeverityBadge(text, colorClass) {
    const badge = document.getElementById('severityBadge');
    if (!badge) return;
    badge.textContent = text;
    const colors = {
        success: { bg: 'rgba(76,175,80,0.15)', color: '#4caf50' },
        warning: { bg: 'rgba(255,235,59,0.15)', color: '#ffeb3b' },
        orange:  { bg: 'rgba(255,152,0,0.15)',  color: '#ff9800' },
        danger:  { bg: 'rgba(244,67,54,0.15)',  color: '#f44336' },
    };
    const c = colors[colorClass] || colors.success;
    badge.style.background = c.bg;
    badge.style.color = c.color;
}

// ============================================================
// Timer
// ============================================================
function startTimer() {
    elapsedSeconds = 0;
    updateTimerDisplay();
    timerInterval = setInterval(() => {
        if (!testPaused) { elapsedSeconds++; updateTimerDisplay(); }
    }, 1000);
}

function stopTimer() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
}

function updateTimerDisplay() {
    const el = document.getElementById('sessionTimer');
    if (el) {
        const min = Math.floor(elapsedSeconds / 60).toString().padStart(2, '0');
        const sec = (elapsedSeconds % 60).toString().padStart(2, '0');
        el.textContent = `${min}:${sec}`;
    }
}

// ============================================================
// Assessment Flow (In-Dashboard)
// ============================================================
async function startTest() {
    const patientIdEl = document.getElementById('patientId');
    const patientNameEl = document.getElementById('patientName');
    
    currentPatientId = patientIdEl ? patientIdEl.textContent : 'PAT-???';
    const patientName = patientNameEl ? patientNameEl.value || 'Unknown' : 'Unknown';

    document.getElementById('btnStartTest').disabled = true;

    try {
        const res = await fetch('/api/start_test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patient_id: currentPatientId, patient_name: patientName }),
        });
        const data = await res.json();
        currentSessionId = data.session_id;

        testRunning = true;
        testPaused = false;
        answeredCount = 0;
        currentQuestionIndex = 0;

        // Update UI
        const modeText = document.getElementById('modeText');
        if (modeText) modeText.textContent = `${data.mode.replace(/_/g, ' ')} – ${data.session_id}`;

        const pb = document.getElementById('questionProgressBar');
        if (pb) pb.style.width = '0%';
        const pt = document.getElementById('questionProgress');
        if (pt) pt.textContent = '0 / 9';

        document.getElementById('btnPauseTest').disabled = false;
        document.getElementById('btnEndTest').disabled = false;

        if (patientNameEl) patientNameEl.disabled = true; // Lock name input

        // Start loops
        startTimer();
        startPolling();
        startFrameCapture();

        // Load first question
        resetQuestionArea();
        loadNextQuestion(0);

    } catch (err) {
        console.error('Failed to start test:', err);
        alert('Failed to start test. Is the server running?');
        document.getElementById('btnStartTest').disabled = false;
    }
}

function resetQuestionArea() {
    const container = document.getElementById('questionsContainer');
    if (container) {
        container.innerHTML = `<div class="modal-q-loading text-center py-4" id="modalQLoading">
            <div class="ai-spinner mb-3"><i class="bi bi-cpu fs-2 text-accent"></i></div>
            <p class="text-muted">AI is preparing your assessment based on live sensor data…</p>
        </div>`;
    }
}

async function loadNextQuestion(index) {
    const loadingEl = document.getElementById('modalQLoading');
    if (loadingEl) loadingEl.style.display = 'block';

    try {
        const res = await fetch('/api/analyze_next_question', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question_index: index }),
        });
        const data = await res.json();

        if (loadingEl) loadingEl.style.display = 'none';

        if (!data.question) {
            // No more questions – show completion
            showComplete();
            return;
        }

        renderQuestion(data.question, !data.has_more);
    } catch (err) {
        console.error('Error loading question:', err);
        if (loadingEl) loadingEl.style.display = 'none';
        
        // Fallback to static list if available
        if (typeof allQuestions !== 'undefined' && index < allQuestions.length) {
            renderQuestion({ ...allQuestions[index], ai_context: null }, index >= allQuestions.length - 1);
        } else {
            showComplete();
        }
    }
}

function renderQuestion(question, isLast) {
    const container = document.getElementById('questionsContainer');
    if (!container) return;

    const sourceClass = question.source === 'AI-Generated' ? 'ai-badge' : '';
    const aiContextHtml = question.ai_context
        ? `<div class="alert alert-secondary border-0 text-white small p-2 mb-3 mt-2" style="background: rgba(255,255,255,0.05);">
               <i class="bi bi-cpu text-accent me-1"></i>${question.ai_context}
           </div>`
        : '';

    container.innerHTML = `
        <div class="question-card" style="display:block;" id="currentQuestionCard">
            <div class="question-number mb-2">
                <span class="q-source-badge ${sourceClass}">${question.source}</span>
                <span class="ms-2 fw-medium text-white">Question ${currentQuestionIndex + 1}</span>
                ${isLast ? '<span class="badge bg-warning ms-2 text-dark">Final Question</span>' : ''}
            </div>
            ${aiContextHtml}
            <p class="question-text fs-5 mb-4">${question.text}</p>
            <div class="question-options">
                ${(question.options || []).map(opt => `
                    <label class="option-label" id="opt-${question.id}-${opt.value}">
                        <input type="radio" name="q_${question.id}" value="${opt.value}"
                               onchange="submitAnswer(${question.id}, '${escapeHtml(question.text)}', ${opt.value}, '${escapeHtml(opt.label)}')">
                        <span class="option-custom-radio"></span>
                        <span class="option-text">${opt.label}</span>
                        <span class="option-score">${opt.value}</span>
                    </label>
                `).join('')}
            </div>
        </div>
    `;
}

function showComplete() {
    const container = document.getElementById('questionsContainer');
    if (container) {
        container.innerHTML = `
            <div class="question-complete text-center py-5">
                <div class="complete-icon mb-3" style="font-size:3rem; color:#4caf50;">
                    <i class="bi bi-check-circle-fill"></i>
                </div>
                <h5>Assessment Complete</h5>
                <p class="text-muted">All questions answered. Click "End & Generate Report" below.</p>
            </div>`;
    }
    const pb = document.getElementById('questionProgressBar');
    if (pb) pb.style.width = '100%';
}

async function submitAnswer(questionId, questionText, answerValue, answerLabel) {
    // Determine the max expected questions (e.g., 9 + 1 adaptive = 10 minimum)
    const expectedTotal = 10; 

    // Disable inputs
    const options = document.querySelectorAll(`input[name="q_${questionId}"]`);
    options.forEach(o => o.disabled = true);

    try {
        const res = await fetch('/api/submit_answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question_id: questionId, question_text: questionText, answer_value: answerValue, answer_label: answerLabel }),
        });
        const data = await res.json();

        answeredCount = data.answered;
        currentQuestionIndex++;

        // Update progress bars
        const pct = (answeredCount / expectedTotal) * 100;
        const pb = document.getElementById('questionProgressBar');
        if (pb) pb.style.width = Math.min(pct, 100) + '%';
        const pt = document.getElementById('questionProgress');
        if (pt) pt.textContent = `${answeredCount} / ${expectedTotal}+`;

        if (data.ai_analysis) {
            showAnswerAnalysis(data.ai_analysis, currentQuestionIndex);
        } else {
            setTimeout(() => loadNextQuestion(currentQuestionIndex), 400);
        }
    } catch (err) {
        console.error('Submit error:', err);
        setTimeout(() => loadNextQuestion(currentQuestionIndex), 400);
    }
}

function showAnswerAnalysis(analysisText, nextIndex) {
    const container = document.getElementById('questionsContainer');
    if (!container) { loadNextQuestion(nextIndex); return; }

    const card = document.createElement('div');
    card.className = 'answer-analysis-card mt-3 p-3 glass-card border-accent-subtle';
    card.innerHTML = `
        <div class="analysis-header fw-bold text-accent mb-2">
            <i class="bi bi-cpu me-2"></i>AI Observation
        </div>
        <p class="analysis-text small mb-3">${analysisText}</p>
        <button class="btn btn-sm btn-outline-light w-100" onclick="this.parentElement.remove(); loadNextQuestion(${nextIndex});">
            Next Question <i class="bi bi-arrow-right ms-1"></i>
        </button>
    `;
    container.appendChild(card);
    card.scrollIntoView({ behavior: 'smooth' });
}

function escapeHtml(str) {
    return (str || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

// ============================================================
// Pause & End Test
// ============================================================
function pauseTest() {
    testPaused = !testPaused;
    const btn = document.getElementById('btnPauseTest');
    if (btn) btn.innerHTML = testPaused
        ? '<i class="bi bi-play-fill me-1"></i>Resume'
        : '<i class="bi bi-pause-fill me-1"></i>Pause';
    const modeText = document.getElementById('modeText');
    if (modeText) modeText.textContent = testPaused ? 'Session paused' : 'Session active';
}

async function endTest() {
    stopTimer();
    stopPolling();
    stopFrameCapture();
    testRunning = false;

    const startBtn = document.getElementById('btnStartTest');
    const pauseBtn = document.getElementById('btnPauseTest');
    const endBtn = document.getElementById('btnEndTest');
    if (startBtn) startBtn.disabled = false;
    if (pauseBtn) pauseBtn.disabled = true;
    if (endBtn) endBtn.disabled = true;

    const modeText = document.getElementById('modeText');
    if (modeText) modeText.textContent = 'Finalizing test…';

    try {
        const res = await fetch('/api/end_test', { method: 'POST' });
        const data = await res.json();

        const resultScore = document.getElementById('resultScore');
        const resultSeverity = document.getElementById('resultSeverity');
        const resultMessage = document.getElementById('resultMessage');
        const resultReportLink = document.getElementById('resultReportLink');

        if (resultScore) resultScore.textContent = Math.round(data.score);
        if (resultSeverity) resultSeverity.textContent = data.severity;
        if (resultMessage) resultMessage.textContent = data.message;
        if (resultReportLink && data.report_url) resultReportLink.href = data.report_url;

        const circle = document.getElementById('resultScoreCircle');
        if (circle) {
            if (data.score <= 4) circle.style.borderColor = '#4caf50';
            else if (data.score <= 9) circle.style.borderColor = '#ffeb3b';
            else if (data.score <= 14) circle.style.borderColor = '#ff9800';
            else circle.style.borderColor = '#f44336';
        }

        // Show modal immediately (non-blocking)
        const resultModal = new bootstrap.Modal(document.getElementById('resultModal'));
        resultModal.show();

        // Poll for report generation status
        if (data.report_generating && data.session_id) {
            pollReportStatus(data.session_id);
        } else {
            hideReportSpinner();
        }

        const pdfBtn = document.getElementById('btnDownloadPdf');
        if (pdfBtn && data.report_url) {
            pdfBtn.style.display = 'inline-block';
            pdfBtn.href = data.report_url;
        }

        if (modeText) modeText.textContent = `Complete – ${data.severity} (${Math.round(data.score)}/27)`;

    } catch (err) {
        console.error('End test error:', err);
        if (modeText) modeText.textContent = 'Error generating report';
        if (endBtn) { endBtn.disabled = false; endBtn.innerHTML = '<i class="bi bi-exclamation-triangle me-1"></i>Retry Report'; }
    }
}

function pollReportStatus(sessionId) {
    const statusEl = document.getElementById('reportStatusText');
    const spinner = document.getElementById('reportSpinner');
    const progressBar = document.getElementById('reportProgressBar');
    const reportLink = document.getElementById('resultReportLink');
    let progress = 30;
    
    // Disable the view report button while generating
    if (reportLink) {
        reportLink.classList.add('disabled');
        reportLink.style.opacity = '0.5';
        reportLink.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Generating...';
    }

    const interval = setInterval(async () => {
        try {
            progress = Math.min(progress + 10, 90);
            if (progressBar) progressBar.style.width = progress + '%';

            const res = await fetch(`/api/report_status/${sessionId}`);
            const data = await res.json();

            if (data.status === 'done') {
                clearInterval(interval);
                if (progressBar) progressBar.style.width = '100%';
                if (statusEl) statusEl.textContent = data.error ? `Report issue: ${data.error}` : 'Report ready!';
                
                // Re-enable and style the report button
                if (reportLink) {
                    reportLink.classList.remove('disabled');
                    reportLink.style.opacity = '1';
                    reportLink.innerHTML = '<i class="bi bi-file-earmark-medical me-1"></i>View Full Report';
                    // Apply success styling when done
                    reportLink.className = 'btn btn-success';
                }
                
                setTimeout(hideReportSpinner, 800);
            }
        } catch (e) {
            clearInterval(interval);
            hideReportSpinner();
            if (reportLink) {
                reportLink.classList.remove('disabled');
                reportLink.style.opacity = '1';
                reportLink.innerHTML = '<i class="bi bi-file-earmark-medical me-1"></i>View Partial Report';
            }
        }
    }, 2000);

    // Timeout after 30s
    setTimeout(() => { 
        clearInterval(interval); 
        hideReportSpinner(); 
        if (reportLink) {
            reportLink.classList.remove('disabled');
            reportLink.style.opacity = '1';
            reportLink.innerHTML = '<i class="bi bi-file-earmark-medical me-1"></i>View Partial Report';
        }
    }, 30000);
}

function hideReportSpinner() {
    const statusDiv = document.getElementById('reportGenStatus');
    if (statusDiv) statusDiv.style.display = 'none';
}

function handleModalClose() {
    hideReportSpinner();
    // Reset test completely and reload the page cleanly
    window.location.reload();
}

// ============================================================
// Camera Vitals Tab Switching
// ============================================================
function switchVitalsTab(btn, tabName) {
    // Deactivate all tabs and contents
    document.querySelectorAll('.vitals-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.vitals-tab-content').forEach(c => c.classList.remove('active'));
    // Activate selected
    btn.classList.add('active');
    const target = document.getElementById('tab-' + tabName);
    if (target) target.classList.add('active');
}

// ============================================================
// Camera Fullscreen Toggle
// ============================================================
let cameraFullscreen = false;
function toggleCameraFullscreen() {
    cameraFullscreen = !cameraFullscreen;
    document.body.classList.toggle('camera-fullscreen', cameraFullscreen);
    const icon = document.querySelector('#cameraExpandIcon i');
    if (icon) {
        icon.className = cameraFullscreen ? 'bi bi-fullscreen-exit' : 'bi bi-arrows-fullscreen';
    }
    // Re-sync overlay after layout change
    setTimeout(syncOverlaySize, 100);
}

// Close fullscreen on Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && cameraFullscreen) toggleCameraFullscreen();
});

// ============================================================
// Live Serial Monitor
// ============================================================
let lastSerialLineCount = 0;

async function updateSerialMonitor() {
    try {
        const res = await fetch('/api/serial_log');
        const data = await res.json();
        const output = document.getElementById('serialOutput');
        const badge = document.getElementById('serialPortBadge');

        if (!output) return;

        // Update connection badge
        if (badge) {
            if (data.connected) {
                badge.className = 'badge bg-success-subtle text-success';
                badge.innerHTML = `<i class="bi bi-check-circle me-1"></i>${data.port}`;
            } else {
                badge.className = 'badge bg-danger-subtle text-danger';
                badge.innerHTML = '<i class="bi bi-x-circle me-1"></i>DISCONNECTED';
            }
        }

        // Only update content if there are new lines
        const lines = data.lines || [];
        if (lines.length !== lastSerialLineCount) {
            lastSerialLineCount = lines.length;
            // Show last 30 lines max for performance
            const displayLines = lines.slice(-30);
            output.textContent = displayLines.length > 0
                ? displayLines.join('\n')
                : (data.connected ? 'Connected. Waiting for data...' : 'ESP32 not connected.');
            // Auto-scroll to bottom
            output.scrollTop = output.scrollHeight;
        }
    } catch (e) {
        // Ignore fetch errors
    }
}
