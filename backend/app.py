"""
NeuroGuard Clinic – Flask Backend (Real Data)
Camera analysis via MediaPipe, Groq LLM, MySQL storage.
No mock/fake data – errors shown clearly on frontend.
"""
import os
import sys
import json
import time
import math
import base64
import uuid
import threading
import traceback
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings
load_dotenv()
# ---------------------------------------------------------------------------
# Azure Blob Storage
# ---------------------------------------------------------------------------
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER = "raw-data"
AZURE_SESSION_PREFIX = "sessions"

azure_blob_service = None

if AZURE_CONNECTION_STRING:
    try:
        azure_blob_service = BlobServiceClient.from_connection_string(
            AZURE_CONNECTION_STRING
        )
        print("[Azure] ✓ Blob Storage connected")

        # Ensure the target container exists (no-op if it already does)
        try:
            container_client = azure_blob_service.get_container_client(AZURE_CONTAINER)
            if not container_client.exists():
                container_client.create_container()
                print(f"[Azure] ✓ Container created: {AZURE_CONTAINER}")
            else:
                print(f"[Azure] ✓ Container ready: {AZURE_CONTAINER}")
        except Exception as e:
            print(f"[Azure] ⚠ Container check failed ({AZURE_CONTAINER}): {e}")

    except Exception as e:
        azure_blob_service = None
        print(f"[Azure] ⚠ Connection failed: {e}")
else:
    print("[Azure] ⚠ AZURE_STORAGE_CONNECTION_STRING not found")


def upload_session_to_azure(session_data):
    """
    Upload one completed NeuroGuard session as JSON to:
        raw-data/sessions/<session_id>.json

    Compact session payload only – never camera frames or landmark streams.
    Any failure is logged and swallowed so the app keeps running normally.
    Returns True on success, False otherwise.
    """
    if azure_blob_service is None:
        print("[Azure] ⚠ Upload skipped – Blob Storage not configured")
        return False

    session_id = None
    try:
        session_id = session_data.get('session_id') or f'UNKNOWN-{uuid.uuid4().hex[:6].upper()}'
        blob_name = f"{AZURE_SESSION_PREFIX}/{session_id}.json"

        payload = json.dumps(session_data, indent=2, default=str)

        blob_client = azure_blob_service.get_blob_client(
            container=AZURE_CONTAINER, blob=blob_name
        )
        blob_client.upload_blob(
            payload.encode('utf-8'),
            overwrite=True,
            content_settings=ContentSettings(content_type='application/json'),
        )

        print(f"[Azure] ✓ Session uploaded: {AZURE_CONTAINER}/{blob_name} "
              f"({len(payload) / 1024:.1f} KB)")
        return True

    except Exception as e:
        print(f"[Azure] ⚠ Session upload failed ({session_id}): {e}")
        return False


# Add parent dir so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import Database
from groq_report import GroqReportGenerator
from models import (
    PHQ9_QUESTIONS, QUESTION_OPTIONS, OPTION_LABELS,
    get_severity, get_severity_color,
)

# --- Load Trained Ensemble Models ---
import pickle

combined_model = None
ensemble_models = {}
meta_learner = None
meta_learner_type = None

try:
    import tensorflow as tf
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

    # Load 5 specialist models (Camera + MAX30102 + PHQ-9)
    specialist_names = ['expression_model', 'blink_model', 'posture_model', 'physio_model', 'phq9_model']
    for name in specialist_names:
        path = os.path.join(MODELS_DIR, f'{name}.h5')
        if os.path.exists(path):
            ensemble_models[name.replace('_model', '')] = tf.keras.models.load_model(path)

    if len(ensemble_models) == 5:
        print(f"[AI] ✓ All 5 specialist models loaded: {list(ensemble_models.keys())}")

        # Load meta-learner
        config_path = os.path.join(MODELS_DIR, 'ensemble_config.json')
        if os.path.exists(config_path):
            import json as _json
            with open(config_path) as f:
                ens_config = _json.load(f)
            meta_learner_type = ens_config.get('meta_type', 'sklearn')

            if meta_learner_type == 'sklearn':
                pkl_path = os.path.join(MODELS_DIR, 'meta_learner.pkl')
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        meta_learner = pickle.load(f)
            else:
                h5_path = os.path.join(MODELS_DIR, 'meta_learner.h5')
                if os.path.exists(h5_path):
                    meta_learner = tf.keras.models.load_model(h5_path)

            if meta_learner is not None:
                print(f"[AI] ✓ Ensemble Meta-Learner loaded ({ens_config.get('best_name')}, "
                      f"accuracy: {ens_config.get('best_accuracy', 0)*100:.1f}%)")
    else:
        print(f"[AI] ⚠ Only {len(ensemble_models)}/5 specialist models found, trying fallback...")

    # Fallback to combined model
    fallback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neuro_model.h5')
    if os.path.exists(fallback_path):
        combined_model = tf.keras.models.load_model(fallback_path)
        print("[AI] ✓ Fallback combined model loaded.")

except Exception as e:
    print(f"[AI] Using base heuristics (model load error: {e})")

# ---------------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend')

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, 'templates'),
    static_folder=os.path.join(FRONTEND_DIR, 'static'),
)
app.secret_key = os.urandom(24).hex()

# ---------------------------------------------------------------------------
# Initialize Services
# ---------------------------------------------------------------------------
db = Database()
groq = GroqReportGenerator()

print(f"[App] Database available: {db.available}")
print(f"[App] Groq available: {groq.is_available()}")


# ---------------------------------------------------------------------------
# Camera Analysis Engine (MediaPipe)
# ---------------------------------------------------------------------------
class CameraAnalyzer:
    """Real-time face + posture analysis using MediaPipe."""

    # Eye landmark indices for EAR blink detection (FaceMesh 478 landmarks)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    # Mouth landmarks for expression estimation
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    LEFT_BROW_INNER = 107
    RIGHT_BROW_INNER = 336
    NOSE_TIP = 4

    EAR_THRESHOLD = 0.21
    BLINK_CONSEC_FRAMES = 2

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Blink tracking
        self.blink_counter = 0
        self.blink_total = 0
        self.ear_below_threshold = 0
        self.blink_timestamps = deque(maxlen=200)

        # Rolling averages (smooth the metrics, reduced to 5 for high real-time responsiveness)
        self.expression_history = deque(maxlen=5)
        self.posture_history = deque(maxlen=5)

        self.last_result = self._empty_result()
        self.frame_count = 0
        self.start_time = time.time()

    def _empty_result(self):
        return {
            'camera_active': False,
            'face_detected': False,
            'blink_rate': 0,
            'facial_expression': {'neutral': 0, 'sad': 0, 'happy': 0, 'anxious': 0},
            'posture_score': 0,
            'landmarks': [],
            'pose_landmarks': [],
        }

    def _eye_aspect_ratio(self, landmarks, eye_indices, w, h):
        """Compute Eye Aspect Ratio (EAR) for blink detection."""
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
        # Vertical distances
        v1 = math.dist(pts[1], pts[5])
        v2 = math.dist(pts[2], pts[4])
        # Horizontal distance
        h1 = math.dist(pts[0], pts[3])
        if h1 == 0:
            return 0.3
        return (v1 + v2) / (2.0 * h1)

    def _estimate_expression(self, landmarks, w, h):
        """Estimate facial expression from landmark geometry."""
        # Mouth openness (happy indicator)
        upper = landmarks[self.UPPER_LIP]
        lower = landmarks[self.LOWER_LIP]
        left_m = landmarks[self.LEFT_MOUTH]
        right_m = landmarks[self.RIGHT_MOUTH]

        mouth_open = abs(upper.y - lower.y) * h
        mouth_width = abs(left_m.x - right_m.x) * w

        # Brow position (anxious/sad indicator)
        left_brow = landmarks[self.LEFT_BROW_INNER]
        right_brow = landmarks[self.RIGHT_BROW_INNER]
        nose = landmarks[self.NOSE_TIP]

        brow_height = ((nose.y - left_brow.y) + (nose.y - right_brow.y)) / 2 * h

        # Mouth corners relative to center (smile detection)
        mouth_center_y = (upper.y + lower.y) / 2
        left_corner_y = landmarks[61].y
        right_corner_y = landmarks[291].y
        corner_avg = (left_corner_y + right_corner_y) / 2
        smile_ratio = (mouth_center_y - corner_avg) * h

        # Normalize to rough percentages
        mouth_ratio = mouth_open / max(mouth_width, 1) if mouth_width > 0 else 0

        happy = max(0, min(100, smile_ratio * 300 + mouth_ratio * 50))
        sad = max(0, min(100, 30 - smile_ratio * 200 + (40 - brow_height) * 2))
        anxious = max(0, min(100, (brow_height - 35) * 5 + mouth_ratio * 80))
        neutral = max(0, 100 - happy - sad - anxious)

        # Ensure they sum to ~100
        total = happy + sad + anxious + neutral
        if total > 0:
            happy = happy / total * 100
            sad = sad / total * 100
            anxious = anxious / total * 100
            neutral = neutral / total * 100

        return {
            'neutral': round(neutral, 1),
            'sad': round(sad, 1),
            'happy': round(happy, 1),
            'anxious': round(anxious, 1),
        }

    def _estimate_posture(self, pose_landmarks, w, h):
        """Estimate posture score from pose landmarks."""
        try:
            lm = pose_landmarks.landmark
            # Shoulder alignment
            left_shoulder = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

            # Head position (nose relative to shoulders)
            nose = lm[mp.solutions.pose.PoseLandmark.NOSE]

            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2

            # Shoulder tilt (should be minimal)
            shoulder_tilt = abs(left_shoulder.y - right_shoulder.y) * h
            tilt_penalty = min(30, shoulder_tilt * 2)

            # Head forward (nose should be above shoulders, not too far forward)
            head_offset = abs(nose.x - shoulder_center_x) * w
            forward_penalty = min(30, head_offset * 1.5)

            # Upright bonus (nose well above shoulders)
            upright = (shoulder_center_y - nose.y) * h
            upright_score = min(40, max(0, upright * 0.8))

            score = max(0, min(100, 60 + upright_score - tilt_penalty - forward_penalty))
            return round(score, 1)
        except Exception:
            return 50.0

    def analyze_frame(self, frame_data_b64):
        """
        Analyze a base64-encoded JPEG frame.
        Returns analysis results + landmark coordinates for overlay.
        """
        try:
            # Decode base64 → numpy array
            if ',' in frame_data_b64:
                frame_data_b64 = frame_data_b64.split(',')[1]

            img_bytes = base64.b64decode(frame_data_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return self._empty_result()

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = {
                'camera_active': True,
                'face_detected': False,
                'blink_rate': 0,
                'facial_expression': {'neutral': 100, 'sad': 0, 'happy': 0, 'anxious': 0},
                'posture_score': 0,
                'landmarks': [],
                'pose_landmarks': [],
            }

            # --- Face Mesh ---
            face_results = self.face_mesh.process(rgb)
            if face_results.multi_face_landmarks:
                face_lm = face_results.multi_face_landmarks[0]
                result['face_detected'] = True

                # EAR for blink detection
                left_ear = self._eye_aspect_ratio(face_lm.landmark, self.LEFT_EYE, w, h)
                right_ear = self._eye_aspect_ratio(face_lm.landmark, self.RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < self.EAR_THRESHOLD:
                    self.ear_below_threshold += 1
                else:
                    if self.ear_below_threshold >= self.BLINK_CONSEC_FRAMES:
                        self.blink_total += 1
                        self.blink_timestamps.append(time.time())
                    self.ear_below_threshold = 0

                # Blink rate (per minute) from last 60 seconds
                now = time.time()
                recent = [t for t in self.blink_timestamps if now - t < 60]
                elapsed = now - self.start_time
                if elapsed > 5:
                    result['blink_rate'] = round(len(recent) * (60 / min(elapsed, 60)), 1)
                else:
                    result['blink_rate'] = 0

                # Expression estimation
                expr = self._estimate_expression(face_lm.landmark, w, h)
                self.expression_history.append(expr)
                # Smoothed average
                avg_expr = {}
                for key in ['neutral', 'sad', 'happy', 'anxious']:
                    vals = [e[key] for e in self.expression_history]
                    avg_expr[key] = round(sum(vals) / len(vals), 1)
                result['facial_expression'] = avg_expr

                # Extract key face landmarks for overlay (every 3rd for performance)
                face_pts = []
                for i, lm in enumerate(face_lm.landmark):
                    if i % 3 == 0:
                        face_pts.append([round(lm.x, 4), round(lm.y, 4)])
                result['landmarks'] = face_pts

            # --- Pose ---
            pose_results = self.pose.process(rgb)
            if pose_results.pose_landmarks:
                posture = self._estimate_posture(pose_results.pose_landmarks, w, h)
                self.posture_history.append(posture)
                avg_posture = sum(self.posture_history) / len(self.posture_history)
                result['posture_score'] = round(avg_posture, 1)

                # Key pose landmarks for overlay
                pose_pts = []
                for lm in pose_results.pose_landmarks.landmark:
                    if lm.visibility > 0.5:
                        pose_pts.append([round(lm.x, 4), round(lm.y, 4)])
                    else:
                        pose_pts.append(None)
                result['pose_landmarks'] = pose_pts

            self.last_result = result
            self.frame_count += 1
            return result

        except Exception as e:
            print(f"[CameraAnalyzer] Frame analysis error: {e}")
            traceback.print_exc()
            return self._empty_result()

    def reset(self):
        """Reset counters for a new test session."""
        self.blink_counter = 0
        self.blink_total = 0
        self.ear_below_threshold = 0
        self.blink_timestamps.clear()
        self.expression_history.clear()
        self.posture_history.clear()
        self.last_result = self._empty_result()
        self.frame_count = 0
        self.start_time = time.time()


# Initialize global camera analyzer
camera_analyzer = CameraAnalyzer()


# ---------------------------------------------------------------------------
# ESP32 Serial Manager (graceful fallback)
# ---------------------------------------------------------------------------
class SerialSensorManager:
    """Manages ESP32 + MAX30102 serial connection with graceful fallback."""

    def __init__(self):
        self.port = os.getenv('SERIAL_PORT', '/dev/cu.usbserial-0001')
        self.baud = 115200
        self.serial_conn = None
        self.last_data = {}
        self.error = None
        self.finger_detected = False
        self.status = 'disconnected'
        self._connected = False
        self.serial_log = deque(maxlen=100)  # Last 100 serial lines for live monitor

        # HRV computation from real beat intervals
        self._beat_times = deque(maxlen=50)
        self._ibi_list = deque(maxlen=50)

        # EMG tracking
        self._emg_history = deque(maxlen=300)  # ~30s of EMG data at 100ms intervals
        self._muscle_active_count = 0
        self._emg_sample_count = 0

        # Reconnection tracking
        self._last_connect_attempt = 0
        self._reconnect_interval = 3  # seconds between reconnect attempts

        self._try_connect()

        # Start background reader thread
        self._reader_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
        self._reader_thread.start()

    def _try_connect(self):
        """Attempt to connect to the ESP32 serial port."""
        self._last_connect_attempt = time.time()
        # Close any existing broken connection first
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None

        try:
            import serial
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=0.5)
            self._connected = True
            self.error = None
            self.status = 'connected'
            print(f"[Serial] ✅ Connected to ESP32 on {self.port}")
        except Exception as e:
            self._connected = False
            self.serial_conn = None
            self.error = f"ESP32 not connected on {self.port}: {str(e)}"
            self.status = 'disconnected'
            print(f"[Serial] {self.error}")

    def _serial_reader_loop(self):
        """Background thread that continuously reads serial data."""
        while True:
            if not self._connected or not self.serial_conn:
                # Try to reconnect periodically
                if time.time() - self._last_connect_attempt > self._reconnect_interval:
                    self._try_connect()
                    # If connected, flush stale data
                    if self._connected and self.serial_conn:
                        try:
                            self.serial_conn.reset_input_buffer()
                        except Exception:
                            pass
                time.sleep(1)
                continue

            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self.serial_log.append(f"[{time.strftime('%H:%M:%S')}] {line}")
                    if line.startswith('DATA:'):
                        self._parse_data_line(line)
                    elif line.startswith('STATUS:'):
                        self._parse_status_line(line)
                else:
                    time.sleep(0.05)
            except (OSError, Exception) as e:
                # USB was physically disconnected
                print(f"[Serial] ❌ ESP32 disconnected: {e}")
                self._connected = False
                self.finger_detected = False
                self.error = f"ESP32 disconnected: {str(e)}"
                self.status = 'disconnected'
                # Keep last_data so report still has sensor readings
                try:
                    self.serial_conn.close()
                except Exception:
                    pass
                self.serial_conn = None
                time.sleep(2)  # Wait before trying to reconnect

    def _compute_hrv(self):
        """Compute HRV metrics (RMSSD, SDNN) from inter-beat intervals."""
        if len(self._ibi_list) < 3:
            return {'rmssd': 0.0, 'sdnn': 0.0}
        ibis = list(self._ibi_list)
        mean_ibi = sum(ibis) / len(ibis)
        sdnn = (sum((x - mean_ibi) ** 2 for x in ibis) / len(ibis)) ** 0.5
        diffs = [(ibis[i+1] - ibis[i]) ** 2 for i in range(len(ibis) - 1)]
        rmssd = (sum(diffs) / len(diffs)) ** 0.5 if diffs else 0.0
        return {'rmssd': round(rmssd, 2), 'sdnn': round(sdnn, 2)}

    def _parse_data_line(self, line):
        """Parse DATA:IR=...,RED=...,HR=...,SPO2=...,EMG=...,MUSCLE=...,BEAT=...,TS=..."""
        try:
            line = line.strip()
            payload = line[5:]  # strip 'DATA:'
            parts = dict(p.split('=') for p in payload.split(',') if '=' in p)

            ir_val = int(parts.get('IR', '0'))
            red_val = int(parts.get('RED', '0'))
            hr = float(parts.get('HR', '0'))
            spo2 = int(parts.get('SPO2', '0'))
            beat = int(parts.get('BEAT', '0'))
            bavg = int(parts.get('BAVG', '0'))
            ts = int(parts.get('TS', '0'))

            # EMG data
            emg_val = int(parts.get('EMG', '0'))
            muscle_active = int(parts.get('MUSCLE', '0'))

            self.finger_detected = ir_val > 50000

            if beat == 1:
                now = time.time()
                if self._beat_times:
                    ibi = (now - self._beat_times[-1]) * 1000
                    if 300 < ibi < 2000:
                        self._ibi_list.append(ibi)
                self._beat_times.append(now)

            # Track EMG statistics
            self._emg_history.append(emg_val)
            self._emg_sample_count += 1
            if muscle_active:
                self._muscle_active_count += 1

            hrv_metrics = self._compute_hrv()

            # Compute EMG statistics
            emg_stats = self._compute_emg_stats()

            self.last_data = {
                'hrv': {
                    'bpm': bavg if bavg > 0 else max(0, min(220, hr)),
                    'rmssd': hrv_metrics['rmssd'],
                    'sdnn': hrv_metrics['sdnn'],
                    'raw_hr': hr,
                },
                'spo2': spo2,
                'ir': ir_val,
                'red': red_val,
                'emg': {
                    'raw': emg_val,
                    'muscle_active': muscle_active,
                    'avg': emg_stats['avg'],
                    'max': emg_stats['max'],
                    'tension_pct': emg_stats['tension_pct'],
                },
                'finger_detected': self.finger_detected,
                'timestamp': ts,
            }
        except Exception:
            pass  # Skip malformed lines

    def _compute_emg_stats(self):
        """Compute EMG statistics from recent history."""
        if not self._emg_history:
            return {'avg': 0, 'max': 0, 'tension_pct': 0.0}
        emg_list = list(self._emg_history)
        avg_emg = sum(emg_list) / len(emg_list)
        max_emg = max(emg_list)
        tension_pct = round((self._muscle_active_count / max(self._emg_sample_count, 1)) * 100, 1)
        return {'avg': round(avg_emg), 'max': max_emg, 'tension_pct': tension_pct}

    def _parse_status_line(self, line):
        """Parse STATUS: lines from ESP32."""
        try:
            payload = line[7:].strip()
            if 'FINGER_DETECTED' in payload:
                self.finger_detected = True
                self.status = 'finger_detected'
            elif 'NO_FINGER' in payload:
                self.finger_detected = False
                self.status = 'no_finger'
            elif 'SENSOR_DISCONNECTED' in payload:
                self.status = 'sensor_disconnected'
        except Exception:
            pass

    def read_sensors(self):
        """Return latest sensor data. Background thread does actual reading."""
        if not self._connected:
            return {
                'error': self.error or 'ESP32 not connected',
                'hrv': None, 'spo2': None, 'emg': None,
                'finger_detected': False,
            }

        if self.last_data:
            return self.last_data
        else:
            return {
                'hrv': {'bpm': 0, 'rmssd': 0, 'sdnn': 0},
                'spo2': 0,
                'emg': {'raw': 0, 'muscle_active': 0, 'avg': 0, 'max': 0, 'tension_pct': 0},
                'finger_detected': self.finger_detected,
            }

    def is_available(self):
        """Live check — returns True only if USB port is actively connected."""
        return self._connected


serial_mgr = SerialSensorManager()


# ---------------------------------------------------------------------------
# Active Test State
# ---------------------------------------------------------------------------
active_test = {
    'running': False,
    'session_id': None,
    'patient_id': None,
    'start_time': None,
    'answers': [],           # list of dicts with question, answer, sensor snapshot, AI analysis
    'questions_asked': [],   # list of all question texts
    'sensor_snapshots': [],  # periodic sensor snapshots during test
    'mode': 'unknown',
    'sensors_available': {},
    'current_question_index': 0,
    'all_questions': [],     # full question list including adaptive ones
}


def _detect_mode():

    """Determine test mode based on available sensors."""
    cam = camera_analyzer.last_result.get('camera_active', False)
    esp32 = serial_mgr.is_available()
    finger = serial_mgr.finger_detected if esp32 else False

    sensors = {
        'camera': {'online': cam, 'label': 'Mac Camera', 'error': None if cam else 'Camera not streaming'},
        'hrv': {'online': esp32, 'label': 'Heart Rate (MAX30102)', 'error': None if esp32 else (serial_mgr.error or 'ESP32 not connected')},
        'spo2': {'online': esp32, 'label': 'SpO₂ (MAX30102)', 'error': None if esp32 else (serial_mgr.error or 'ESP32 not connected')},
        'emg': {'online': esp32, 'label': 'EMG (Muscle)', 'error': None if esp32 else (serial_mgr.error or 'ESP32 not connected')},
    }

    if cam and esp32:
        mode = 'full_multimodal'
    elif cam:
        mode = 'camera_only'
    elif esp32:
        mode = 'sensor_only'
    else:
        mode = 'questions_only'

    sensors_used = []
    if cam:
        sensors_used.append('Camera')
    if esp32:
        sensors_used.extend(['HR', 'SpO2', 'EMG'])

    return mode, sensors, sensors_used


# ---------------------------------------------------------------------------
# Page Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    """Landing page."""
    sessions = db.get_all_sessions()
    total = len(sessions)
    avg_conf = 0
    if total > 0:
        confs = [s.get('confidence', 0) for s in sessions if s.get('confidence')]
        avg_conf = round(sum(confs) / len(confs), 1) if confs else 0

    mode, sensors, sensors_used = _detect_mode()
    stats = {
        'total_sessions': total,
        'avg_confidence': avg_conf,
        'sensors_active': sum(1 for s in sensors.values() if s['online']),
    }
    return render_template('index.html', stats=stats)


@app.route('/dashboard')
def dashboard():
    """Main live test dashboard."""
    patient_id = request.args.get('patient_id', f'PAT-{uuid.uuid4().hex[:6].upper()}')

    # Build PHQ-9 question list
    questions = []
    for i, q in enumerate(PHQ9_QUESTIONS):
        questions.append({
            'id': i,
            'text': q,
            'options': QUESTION_OPTIONS,
            'source': 'PHQ-9',
        })

    # Detect sensors
    mode, sensor_status, sensors_used = _detect_mode()

    return render_template('dashboard.html',
                           patient_id=patient_id,
                           questions=questions,
                           sensor_status=sensor_status,
                           question_options=QUESTION_OPTIONS,
                           groq_available=groq.is_available(),
                           mode=mode)


@app.route('/report/<session_id>')
def report(session_id):
    """Full report page for a session."""
    session = db.get_session(session_id)

    if not session:
        flash('Session not found', 'danger')
        return redirect(url_for('history'))

    report_data = session.get('report', {})
    answers = session.get('answers', [])
    sensor_summary = session.get('sensor_summary', {})

    severity_color = get_severity_color(session.get('severity', 'Normal'))

    return render_template('report.html',
                           session=session,
                           report_data=report_data,
                           answers=answers,
                           sensor_summary=sensor_summary,
                           severity_color=severity_color)


@app.route('/history')
def history():
    """Past sessions history page."""
    sessions = db.get_all_sessions()
    return render_template('history.html', sessions=sessions)

@app.route('/api/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a specific session by ID."""
    success = db.delete_session(session_id)
    if success:
        return jsonify({'status': 'deleted'})
    return jsonify({'error': 'Failed to delete session'}), 500



@app.route('/settings')
def settings():
    """Settings page."""
    current_settings = {
        'groq_api_key': '••••' + (os.getenv('GROQ_API_KEY', '')[-4:] if os.getenv('GROQ_API_KEY') else ''),
        'groq_available': groq.is_available(),
        'groq_error': groq.error,
        'serial_port': os.getenv('SERIAL_PORT', '/dev/cu.usbserial-0001'),
        'baud_rate': int(os.getenv('BAUD_RATE', '115200')),
        'serial_available': serial_mgr.is_available(),
        'serial_error': serial_mgr.error,
        'db_available': db.available,
        'camera_index': os.getenv('CAMERA_INDEX', '0'),
        'test_duration': int(os.getenv('TEST_DURATION', '180')),
        'enable_camera': os.getenv('ENABLE_CAMERA', 'true').lower() == 'true',
        'enable_eeg': os.getenv('ENABLE_EEG', 'true').lower() == 'true',
        'enable_emg': os.getenv('ENABLE_EMG', 'true').lower() == 'true',
        'enable_hrv': os.getenv('ENABLE_HRV', 'true').lower() == 'true',
        'enable_spo2': os.getenv('ENABLE_SPO2', 'true').lower() == 'true',
        'enable_adaptive_questions': os.getenv('ENABLE_ADAPTIVE_QUESTIONS', 'true').lower() == 'true',
    }
    return render_template('settings.html', settings=current_settings)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.route('/api/settings', methods=['POST'])
def api_settings():
    """Save settings by updating the .env file and reloading config."""
    import dotenv
    data = request.get_json() or {}
    env_file = os.path.join(BASE_DIR, '.env')

    if not os.path.exists(env_file):
        open(env_file, 'w').close()

    try:
        if 'groq_api_key' in data and not data['groq_api_key'].startswith('••••'):
            dotenv.set_key(env_file, 'GROQ_API_KEY', data['groq_api_key'])
            groq.api_key = data['groq_api_key']
            groq._init_client()

        if 'serial_port' in data:
            dotenv.set_key(env_file, 'SERIAL_PORT', data['serial_port'])
        if 'camera_index' in data:
            dotenv.set_key(env_file, 'CAMERA_INDEX', str(data['camera_index']))
        if 'baud_rate' in data:
            dotenv.set_key(env_file, 'BAUD_RATE', str(data['baud_rate']))
        if 'test_duration' in data:
            dotenv.set_key(env_file, 'TEST_DURATION', str(data['test_duration']))

        for key in ['enable_camera', 'enable_eeg', 'enable_emg', 'enable_hrv', 'enable_spo2', 'enable_adaptive_questions']:
            if key in data:
                dotenv.set_key(env_file, key.upper(), 'true' if data[key] else 'false')

        # Since Python takes a while to hot-reload, we update environment immediately for some
        os.environ['SERIAL_PORT'] = data.get('serial_port', os.getenv('SERIAL_PORT', ''))
        os.environ['CAMERA_INDEX'] = str(data.get('camera_index', os.getenv('CAMERA_INDEX', '0')))
        os.environ['BAUD_RATE'] = str(data.get('baud_rate', os.getenv('BAUD_RATE', '115200')))

        for key in ['enable_camera', 'enable_eeg', 'enable_emg', 'enable_hrv', 'enable_spo2', 'enable_adaptive_questions']:
            if key in data:
                os.environ[key.upper()] = 'true' if data[key] else 'false'

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/camera_frame', methods=['POST'])
def api_camera_frame():
    """
    Receive a base64 JPEG camera frame, analyze with MediaPipe.
    Returns analysis results + landmark coordinates for overlay.
    """
    data = request.get_json()
    frame_b64 = data.get('frame', '')

    if not frame_b64:
        return jsonify({'error': 'No frame data received'}), 400

    # Strip data URL prefix if present
    if ',' in frame_b64:
        frame_b64 = frame_b64.split(',', 1)[1]

    result = camera_analyzer.analyze_frame(frame_b64)
    return jsonify(result)


@app.route('/api/serial_log')
def api_serial_log():
    """Return latest serial monitor lines from ESP32."""
    return jsonify({
        'connected': serial_mgr.is_available(),
        'port': serial_mgr.port,
        'finger': serial_mgr.finger_detected,
        'lines': list(serial_mgr.serial_log),
    })

def compute_fusion_score(cam, serial_data, q_total):
    """
    Compute final score from 0-27 based on ML ensemble predictions.
    If ML fails or is unavailable, use a heuristic approach based on q_total + sensor deltas.
    """
    try:
        import numpy as np
        if ensemble_models and meta_learner:
            # Extract features matching train_model.py format
            sad = float(cam.get('facial_expression', {}).get('sad', 0.0))
            blink = float(cam.get('blink_rate', 16.0))
            posture = float(cam.get('posture_score', 88.0))

            hr = float(serial_data.get('hrv', {}).get('bpm', 70.0) if type(serial_data.get('hrv')) is dict else 70.0)
            spo2 = float(serial_data.get('spo2', 98.0))
            rmssd = float(serial_data.get('hrv', {}).get('rmssd', 50.0) if type(serial_data.get('hrv')) is dict else 50.0)

            baseline_q = float(q_total)

            # Raw feature vector
            X = np.array([[sad, blink, posture, hr, spo2, rmssd, baseline_q]])

            # Normalize matching normalize_features()
            scales = np.array([100.0, 60.0, 100.0, 180.0, 100.0, 150.0, 27.0])
            X_n = X / scales

            # Get sub-model predictions
            X_expr = np.column_stack([X_n[:, 0], X_n[:, 0]**2, np.abs(X_n[:, 0] - 0.5)])
            expr_preds = ensemble_models['expression'].predict(X_expr, verbose=0)

            X_blink = np.column_stack([X_n[:, 1], X_n[:, 1]**2, np.abs(X_n[:, 1] - 0.27)])
            blink_preds = ensemble_models['blink'].predict(X_blink, verbose=0)

            X_post = np.column_stack([X_n[:, 2], X_n[:, 2]**2, np.abs(X_n[:, 2] - 0.70)])
            post_preds = ensemble_models['posture'].predict(X_post, verbose=0)

            X_phy = X_n[:, [3, 4, 5]]
            X_phy = np.column_stack([
                X_phy,
                X_phy[:, 0] * X_phy[:, 2],
                X_phy[:, 0] / (X_phy[:, 2] + 0.01),
                X_phy[:, 1] * X_phy[:, 2],
                np.abs(X_phy[:, 0] - 0.42),
            ])
            physio_preds = ensemble_models['physio'].predict(X_phy, verbose=0)

            X_phq = np.column_stack([
                X_n[:, 6], X_n[:, 6]**2,
                (X_n[:, 6] > 0.185).astype(float),
                (X_n[:, 6] > 0.37).astype(float),
                (X_n[:, 6] > 0.74).astype(float),
            ])
            phq_preds = ensemble_models['phq9'].predict(X_phq, verbose=0)

            meta_features = np.column_stack([
                expr_preds, blink_preds, post_preds, physio_preds, phq_preds, X_n
            ])

            if meta_learner_type == 'sklearn':
                probs = meta_learner.predict_proba(meta_features)[0]
            else:
                probs = meta_learner.predict(meta_features, verbose=0)[0]

            # Map probabilities to a 0-27 continuous score
            score = probs[0] * 3 + probs[1] * 12 + probs[2] * 22
            return int(round(max(0, min(27, (score * 0.7 + baseline_q * 0.3)))))

        elif combined_model:
            sad = float(cam.get('facial_expression', {}).get('sad', 0.0))
            blink = float(cam.get('blink_rate', 16.0))
            posture = float(cam.get('posture_score', 88.0))
            hr = float(serial_data.get('hrv', {}).get('bpm', 70.0) if type(serial_data.get('hrv')) is dict else 70.0)
            spo2 = float(serial_data.get('spo2', 98.0))
            rmssd = float(serial_data.get('hrv', {}).get('rmssd', 50.0) if type(serial_data.get('hrv')) is dict else 50.0)
            baseline_q = float(q_total)

            X = np.array([[sad, blink, posture, hr, spo2, rmssd, baseline_q]])
            scales = np.array([100.0, 60.0, 100.0, 180.0, 100.0, 150.0, 27.0])
            X_n = X / scales
            X_ext = np.column_stack([
                X_n,
                X_n[:, 0] * X_n[:, 2],
                X_n[:, 3] / (X_n[:, 5] + 0.01),
                X_n[:, 3] * X_n[:, 5],
                X_n[:, 0] * X_n[:, 6],
            ])
            probs = combined_model.predict(X_ext, verbose=0)[0]
            score = probs[0] * 3 + probs[1] * 12 + probs[2] * 22
            return max(0, min(27, (score * 0.7 + baseline_q * 0.3)))

    except Exception as e:
        print(f"[ML Fusion] Error evaluating model: {e}")

    # Fallback to pure heuristics (e.g. if models not available or error)
    score = float(q_total)

    # 1. Camera adjustments
    sadness = cam.get('facial_expression', {}).get('sad', 0)
    if sadness > 50: score += 2.0
    elif sadness > 30: score += 1.0

    posture = cam.get('posture_score', 100)
    if posture < 40: score += 1.5
    elif posture < 60: score += 0.5

    # 2. Sensor adjustments
    hr = serial_data.get('hrv', {}).get('bpm', 70) if type(serial_data.get('hrv')) is dict else 70
    if hr > 95: score += 1.0

    rmssd = serial_data.get('hrv', {}).get('rmssd', 60) if type(serial_data.get('hrv')) is dict else 60
    if rmssd < 20 and rmssd > 0: score += 2.0
    elif rmssd < 40 and rmssd > 0: score += 1.0

    return max(0, min(27, score))



@app.route('/api/sensor_data')
def api_sensor_data():
    """
    Return current real sensor data (camera analysis + serial sensors).
    No mock/random data. Returns error status for offline sensors.
    """
    cam = camera_analyzer.last_result
    serial_data = serial_mgr.read_sensors()

    mode, sensors, sensors_used = _detect_mode()

    esp32_connected = serial_mgr.is_available()

    response = {
        'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3],
        'mode': mode,
        'sensors': sensors,
        'esp32_connected': esp32_connected,
        'camera': {
            'active': cam.get('camera_active', False),
            'face_detected': cam.get('face_detected', False),
            'blink_rate': cam.get('blink_rate', 0),
            'facial_expression': cam.get('facial_expression', {'neutral': 0, 'sad': 0, 'happy': 0, 'anxious': 0}),
            'posture_score': cam.get('posture_score', 0),
        },
        'hrv': serial_data.get('hrv') if not serial_data.get('error') else None,
        'spo2': serial_data.get('spo2') if not serial_data.get('error') else None,
        'emg': serial_data.get('emg') if not serial_data.get('error') else None,
        'finger_detected': serial_data.get('finger_detected', False),
        'sensor_error': serial_data.get('error'),
        'fusion_score': 0,
        'confidence': 0,
    }

    # Compute fusion score from available data
    if active_test['running']:
        q_total = sum(a.get('answer_value', 0) for a in active_test['answers']) if active_test['answers'] else 0
        response['fusion_score'] = compute_fusion_score(cam, serial_mgr.last_data if esp32_connected else {}, q_total)

        # Confidence based on available sources
        available_count = sum([
            1 if cam.get('face_detected') else 0,
            1 if serial_mgr.is_available() else 0,
            1 if len(active_test['answers']) > 0 else 0,
        ])
        response['confidence'] = round(max(50, min(98, 60 + available_count * 12 + len(active_test['answers']) * 2)), 1)

    return jsonify(response)


@app.route('/api/start_test', methods=['POST'])
def api_start_test():
    """Start a new test session."""
    data = request.get_json() or {}
    patient_id = data.get('patient_id', f'PAT-{uuid.uuid4().hex[:6].upper()}')
    patient_name = data.get('patient_name', 'Unknown')

    camera_analyzer.reset()
    mode, sensors, sensors_used = _detect_mode()

    session_id = f'NG-{datetime.now().strftime("%Y%m%d")}-{uuid.uuid4().hex[:5].upper()}'

    active_test.update({
        'running': True,
        'session_id': session_id,
        'patient_id': patient_id,
        'patient_name': patient_name,
        'start_time': time.time(),
        'answers': [],
        'questions_asked': [PHQ9_QUESTIONS[0]],

        'sensor_snapshots': [],
        'mode': mode,
        'sensors_available': sensors,
        'current_question_index': 0,
        'all_questions': [{'id': i, 'text': q, 'source': 'PHQ-9'} for i, q in enumerate(PHQ9_QUESTIONS)],
    })

    return jsonify({
        'status': 'started',
        'session_id': session_id,
        'patient_id': patient_id,
        'patient_name': patient_name,
        'mode': mode,
        'sensors': sensors,
        'sensors_used': sensors_used,
        'message': f'Test started in {mode.replace("_", " ")} mode',
    })


@app.route('/api/analyze_next_question', methods=['POST'])
def api_analyze_next_question():
    """
    Before showing the next question, analyze sensor data to personalize it.
    If Groq is available and we're past question 4, generate an adaptive question.
    Otherwise return the next PHQ-9 question.
    """
    data = request.get_json() or {}
    question_index = data.get('question_index', 0)

    sensor_snapshot = camera_analyzer.last_result.copy()

    # If groq is not available, just use static PHQ-9
    if not groq.is_available():
        if question_index < len(PHQ9_QUESTIONS):
            return jsonify({
                'question': {
                    'id': question_index,
                    'text': PHQ9_QUESTIONS[question_index],
                    'options': QUESTION_OPTIONS,
                    'source': 'PHQ-9',
                    'ai_context': None,
                },
                'has_more': question_index < len(PHQ9_QUESTIONS) - 1,
            })
        else:
            return jsonify({'question': None, 'has_more': False})

    # Groq is available. We'll ask 4 standard PHQ-9 questions, then generate adaptive ones up to 9 total.
    if question_index < 4:
        return jsonify({
            'question': {
                'id': question_index,
                'text': PHQ9_QUESTIONS[question_index],
                'options': QUESTION_OPTIONS,
                'source': 'PHQ-9',
                'ai_context': None,
            },
            'has_more': True,
        })
    elif question_index < 9:
        prev_answers = {a['question']: a['answer_label'] for a in active_test['answers']}
        ai_q = groq.generate_adaptive_question(
            sensor_data={
                'camera': sensor_snapshot,
                'eeg': serial_mgr.read_sensors().get('eeg', {}),
                'hrv': serial_mgr.read_sensors().get('hrv', {}),
            },
            previous_answers=prev_answers,
            question_index=question_index + 1,
        )

        if 'error' not in ai_q:
            return jsonify({
                'question': {
                    'id': question_index,
                    'text': ai_q['question'],
                    'options': QUESTION_OPTIONS,
                    'source': 'AI-Generated',
                    'ai_context': ai_q.get('reasoning', ''),
                },
                'has_more': question_index < 10,
            })

    # No more questions
    return jsonify({
        'question': None,
        'has_more': False,
    })


@app.route('/api/submit_answer', methods=['POST'])
def api_submit_answer():
    """Submit an answer with sensor snapshot. Generates per-question AI analysis."""
    data = request.get_json()
    question_id = data.get('question_id')
    question_text = data.get('question_text', '')
    answer_value = data.get('answer_value', 0)
    answer_label = OPTION_LABELS.get(answer_value, 'Unknown')

    # Capture sensor snapshot at answer time
    sensor_snapshot = {
        'camera': camera_analyzer.last_result.copy(),
        'serial': serial_mgr.read_sensors(),
        'timestamp': datetime.now().isoformat(),
    }

    # AI analysis of this specific answer
    ai_analysis = ""
    if groq.is_available():
        ai_analysis = groq.analyze_question_response(
            question_text=question_text,
            answer_value=answer_value,
            answer_label=answer_label,
            sensor_snapshot=sensor_snapshot,
        )

    answer_record = {
        'question_id': question_id,
        'question': question_text,
        'answer_value': answer_value,
        'answer_label': answer_label,
        'ai_analysis': ai_analysis,
        'sensor_snapshot': {
            'blink_rate': sensor_snapshot['camera'].get('blink_rate', 0),
            'facial_expression': sensor_snapshot['camera'].get('facial_expression', {}),
            'posture_score': sensor_snapshot['camera'].get('posture_score', 0),
            'face_detected': sensor_snapshot['camera'].get('face_detected', False),
        },
    }

    active_test['answers'].append(answer_record)
    answered_count = len(active_test['answers'])

    return jsonify({
        'status': 'recorded',
        'answered': answered_count,
        'ai_analysis': ai_analysis,
        'groq_available': groq.is_available(),
    })


# Store pending report generation results
_pending_reports = {}

@app.route('/api/end_test', methods=['POST'])
def api_end_test():
    """End test, compute final score, kick off async Groq report generation."""
    if not active_test['running']:
        return jsonify({'error': 'No active test'}), 400

    active_test['running'] = False
    elapsed = time.time() - (active_test['start_time'] or time.time())

    # Calculate individual components
    cam = camera_analyzer.last_result
    serial_data = serial_mgr.last_data if serial_mgr.is_available() else {}
    q_total = sum(a.get('answer_value', 0) for a in active_test['answers']) if active_test['answers'] else 0

    # ML FUSION SCORE
    total_score = compute_fusion_score(cam, serial_data, q_total)
    severity = get_severity(int(round(total_score)))

    # Confidence
    mode, sensors, sensors_used = _detect_mode()
    available_count = sum(1 for s in sensors.values() if s['online'])
    confidence = round(max(50, min(98, 60 + available_count * 10 + len(active_test['answers']) * 2)), 1)

    # Build sensor summary with EMG data
    cam = camera_analyzer.last_result
    emg_data = serial_data.get('emg', {})
    emg_stats = serial_mgr._compute_emg_stats() if serial_mgr.is_available() else {}

    sensor_summary = {
        'mode': active_test['mode'],
        'camera': {
            'active': cam.get('camera_active', False),
            'face_detected': cam.get('face_detected', False),
            'avg_blink_rate': cam.get('blink_rate', 0),
            'dominant_expression': max(cam.get('facial_expression', {'neutral': 100}),
                                       key=cam.get('facial_expression', {'neutral': 100}).get),
            'facial_expression': cam.get('facial_expression', {}),
            'avg_posture_score': cam.get('posture_score', 0),
            'total_frames_analyzed': camera_analyzer.frame_count,
        },
        'emg': {
            'available': serial_mgr.is_available(),
            'current_value': emg_data.get('raw', 0) if isinstance(emg_data, dict) else 0,
            'avg_value': emg_stats.get('avg', 0),
            'max_value': emg_stats.get('max', 0),
            'muscle_tension_pct': emg_stats.get('tension_pct', 0),
            'total_samples': serial_mgr._emg_sample_count,
        },
        'eeg_available': serial_mgr.is_available(),
        'eeg_error': serial_mgr.error,
    }

    session_id = active_test['session_id']
    duration_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    # Save initial session (without report) immediately
    session_data = {
        'session_id': session_id,
        'patient_id': active_test['patient_id'],
        'patient_name': active_test.get('patient_name', 'Unknown'),
        'score': total_score,
        'severity': severity,
        'confidence': confidence,
        'mode': active_test['mode'],
        'sensors_used': sensors_used,
        'duration_seconds': int(elapsed),
        'answers': active_test['answers'],
        'sensor_summary': sensor_summary,
        'report': {},
    }

    # Archive the completed session to Azure Blob Storage.
    # Never blocks or breaks the request – failures are logged inside the helper.
    upload_session_to_azure(session_data)

    # Generate Groq report in background thread to prevent UI freeze
    _pending_reports[session_id] = {'status': 'generating', 'report': {}, 'error': None}

    def _generate_report_async(sid, score, sev, conf, ss, answers, md, dur):
        try:
            if groq.is_available():
                report = groq.generate_report(
                    score=score, severity=sev, confidence=conf,
                    sensor_summary=ss, answers_with_analysis=answers,
                    mode=md, duration=dur,
                )
                if 'error' in report:
                    _pending_reports[sid] = {'status': 'done', 'report': {}, 'error': report['error']}
                else:
                    _pending_reports[sid] = {'status': 'done', 'report': report, 'error': None}
            else:
                _pending_reports[sid] = {'status': 'done', 'report': {}, 'error': groq.error}

            # Update session in DB with the generated report
            session_data['report'] = _pending_reports[sid].get('report', {})
            db.save_session(session_data)
            # Re-upload so the archived blob also carries the finished report
            upload_session_to_azure(session_data)
        except Exception as e:
            _pending_reports[sid] = {'status': 'done', 'report': {}, 'error': str(e)}
            session_data['report'] = {}
            db.save_session(session_data)

    report_thread = threading.Thread(
        target=_generate_report_async,
        args=(session_id, total_score, severity, confidence, sensor_summary,
              active_test['answers'], active_test['mode'], duration_str),
        daemon=True
    )
    report_thread.start()

    return jsonify({
        'status': 'completed',
        'session_id': session_id,
        'score': total_score,
        'severity': severity,
        'confidence': confidence,
        'mode': active_test['mode'],
        'report_url': f'/report/{session_id}',
        'report_generating': True,
        'message': f'Test complete. Severity: {severity} ({total_score}/27)',
    })


@app.route('/api/report_status/<session_id>')
def api_report_status(session_id):
    """Check if the async report generation is complete."""
    pending = _pending_reports.get(session_id)
    if pending:
        return jsonify({
            'status': pending['status'],
            'error': pending.get('error'),
            'report_url': f'/report/{session_id}' if pending['status'] == 'done' else None,
        })
    # If not in pending, it might already be saved
    session = db.get_session(session_id)
    if session and session.get('report'):
        return jsonify({'status': 'done', 'error': None, 'report_url': f'/report/{session_id}'})
    return jsonify({'status': 'unknown', 'error': 'Session not found'})





@app.route('/api/status')
def api_status():
    """System status check."""
    return jsonify({
        'database': db.available,
        'groq': groq.is_available(),
        'groq_error': groq.error,
        'serial': serial_mgr.is_available(),
        'serial_error': serial_mgr.error,
        'camera_frames': camera_analyzer.frame_count,
        'test_running': active_test['running'],
        'azure': azure_blob_service is not None,
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  NeuroGuard Clinic – Starting Server (REAL DATA MODE)")
    print(f"  Database: {'✅ MySQL' if db.available else '⚠️  In-memory fallback'}")
    print(f"  Groq LLM: {'✅ Connected' if groq.is_available() else '❌ ' + (groq.error or 'Not configured')}")
    print(f"  ESP32:    {'✅ Connected' if serial_mgr.is_available() else '⚠️  Not connected (camera-only mode)'}")
    print(f"  Azure:    {'✅ Blob Storage (' + AZURE_CONTAINER + ')' if azure_blob_service else '⚠️  Not configured'}")
    print(f"  Open http://127.0.0.1:5001 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5001)