# NeuroGuard Clinic – Backend Rewrite (Mock → Real Data)

Replace all mock/random data with real camera analysis (MediaPipe), real Groq LLM integration, MySQL database storage, and proper error handling. If camera or sensors are unavailable, show clear error indicators on the frontend instead of fake data.

## User Review Required

> [!IMPORTANT]
> **Camera Analysis Architecture**: Camera frames will be captured in the browser, sent to the Flask backend as base64 JPEG via `/api/camera_frame`, and analyzed server-side with MediaPipe FaceMesh + Pose. Results (blink rate, expression, posture, face landmarks) are sent back to the browser for overlay rendering. This means MediaPipe runs on the backend (Python) and the browser just displays results.

> [!WARNING]
> **ESP32 Serial**: Since you likely don't have the ESP32+BioAmp connected right now, the serial sensor code will have graceful fallback – if no serial device is detected, the system runs in `camera_only` mode automatically. EEG/EMG/HRV charts will show "Sensor offline" instead of fake data.

> [!IMPORTANT]
> **Groq API Key**: Your [.env](file:///Users/eashanjain/Documents/Minor-eeg-project/.env) has `grok api=gsk_IheJnIGwm...` which needs to be renamed to `GROQ_API_KEY=gsk_IheJnIGwm...`. I'll fix this.

---

## Proposed Changes

### Environment Config

#### [MODIFY] [.env](file:///Users/eashanjain/Documents/Minor-eeg-project/.env)
- Fix `grok api=...` → `GROQ_API_KEY=...`
- Add `SERIAL_PORT` and `CAMERA_INDEX` config

---

### Backend Core

#### [MODIFY] [groq_report.py](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/groq_report.py)
- Full rewrite: real Groq API calls using `groq` Python SDK
- [generate_adaptive_question()](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/groq_report.py#10-19) – sends live sensor data to Groq, returns 1-2 adaptive questions as JSON
- [generate_report()](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/groq_report.py#20-26) – sends final score + sensor insights + answers to Groq, returns HTML report
- Error handling: if Groq API fails, returns error message string instead of crashing

#### [MODIFY] [database.py](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/database.py)
- Expand [sessions](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/database.py#74-87) table with proper columns: `session_id`, `patient_id`, `score`, [severity](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/app.py#135-147), `confidence`, `mode`, `sensors_used`, `duration`, `report_html`, `answers_json`, `sensor_data_json`
- Add [save_session()](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/database.py#45-59) / [get_session()](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/database.py#60-73) / [get_all_sessions()](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/database.py#74-87) / [delete_session()](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/database.py#88-101) methods using structured data
- Graceful fallback: if MySQL is unavailable, log error and continue (sessions saved in-memory)

#### [MODIFY] [app.py](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/app.py)
**This is the largest change.** Complete rewrite replacing mock data with real systems:

1. **Camera Analysis Engine** (new class `CameraAnalyzer`):
   - MediaPipe FaceMesh: detect 478 face landmarks, compute Eye Aspect Ratio (EAR) for blink detection
   - MediaPipe Pose: detect 33 body landmarks, compute posture score (shoulder alignment, head tilt)
   - Facial expression estimation from landmark geometry (mouth openness, brow position, etc.)
   - Maintains rolling averages for smooth metrics
   - Returns face landmarks as JSON for browser overlay rendering

2. **New API `/api/camera_frame` (POST)**:
   - Receives base64 JPEG frame from browser
   - Processes with `CameraAnalyzer`
   - Returns: blink count, expression scores, posture score, face landmark coordinates

3. **ESP32 Serial Manager** (new class `SerialSensorManager`):
   - Attempts to open serial port from [.env](file:///Users/eashanjain/Documents/Minor-eeg-project/.env) config
   - If unavailable: sets `sensor_mode = "camera_only"`, returns error status
   - If available: reads `BIOAMP:xxx,HR:yyy` lines, parses EEG/HRV values

4. **Updated `/api/sensor_data`**:
   - Returns real camera analysis data (from latest processed frame)
   - Returns real serial sensor data (if connected) or error status
   - Never returns fake/random data

5. **Updated `/api/start_test`**:
   - Detects which sensors are available (camera, serial)
   - Sets mode: `full_multimodal`, `camera_only`, `sensor_only`, or `questions_only`
   - Returns sensor status with error messages for offline sensors

6. **Updated `/api/submit_answer`**:
   - Calls Groq to generate adaptive question (with real sensor data context)
   - If Groq fails, returns static fallback question with error indicator

7. **Updated `/api/end_test`**:
   - Computes final score from real sensor analysis + question answers
   - Calls Groq to generate full report (with real data)
   - Saves session to MySQL
   - Returns real report URL

8. **Updated `/report/<session_id>`**:
   - Loads real session from database
   - Shows real Groq-generated report
   - If session not found, shows error page

9. **Updated `/history`**:
   - Loads sessions from MySQL database

#### [MODIFY] [models.py](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/models.py)
- Add proper data classes matching the database schema

---

### Frontend Updates

#### [MODIFY] [dashboard.js](file:///Users/eashanjain/Documents/Minor-eeg-project/frontend/static/js/dashboard.js)
- Capture camera frames from `<video>` element → draw to hidden canvas → convert to base64 JPEG
- Send frames to `/api/camera_frame` every 500ms
- Draw received face landmarks on `<canvas id="cameraOverlay">` (green dots + mesh lines)
- Show real error messages when sensors return error status
- Mirror camera feed for natural user experience

#### [MODIFY] [dashboard.html](file:///Users/eashanjain/Documents/Minor-eeg-project/frontend/templates/dashboard.html)
- Add "Test Instructions" panel (collapsible) with step-by-step guidance
- Add error message areas for each sensor

#### [MODIFY] [style.css](file:///Users/eashanjain/Documents/Minor-eeg-project/frontend/static/css/style.css)
- Add styles for face landmark overlay, instruction panel, error indicators

---

### Requirements

#### [MODIFY] [requirements.txt](file:///Users/eashanjain/Documents/Minor-eeg-project/requirements.txt)
- Add `python-dotenv` (for [.env](file:///Users/eashanjain/Documents/Minor-eeg-project/.env) loading)

---

## Verification Plan

### Automated Tests
No existing test suite found. Given the scope (hardware sensors, camera, LLM API), automated unit tests are not practical for this project.

### Manual Verification (Browser Testing)
I will use the browser tool to verify end-to-end:

1. **Start Flask server**: `cd /Users/eashanjain/Documents/Minor-eeg-project && python backend/app.py`
2. **Open dashboard**: Navigate to `http://127.0.0.1:5001/dashboard`
3. **Verify camera feed**: Check that Mac camera activates, face landmarks are drawn as green overlay dots
4. **Verify sensor status**: EEG/EMG/HRV sensors should show "Offline" (red) since no ESP32 is connected
5. **Verify fallback message**: Dashboard should show "Camera-only mode – EEG sensor offline"
6. **Start test**: Click "Start Test", verify timer starts and camera analysis metrics update with real data
7. **Answer questions**: Submit PHQ-9 answers, verify adaptive question appears (Groq-generated or fallback)
8. **End test**: Click "End & Generate Report", verify Groq generates actual report text
9. **Check report page**: Navigate to report, verify real sensor insights and LLM text (not hardcoded)
10. **Check history**: Navigate to history, verify session was saved to MySQL
