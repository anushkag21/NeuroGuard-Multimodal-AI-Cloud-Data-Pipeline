# NeuroGuard Clinic – Backend Rewrite (Real Data)

## Planning
- [x] Explore existing codebase and understand current state
- [/] Create implementation plan
- [ ] Get user approval

## Execution

### 1. Fix [.env](file:///Users/eashanjain/Documents/Minor-eeg-project/.env) File
- [ ] Fix Groq API key format (`GROQ_API_KEY=...`)

### 2. Rewrite [backend/groq_report.py](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/groq_report.py) – Real Groq Integration
- [ ] Implement [generate_adaptive_question()](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/groq_report.py#10-19) with live Groq API
- [ ] Implement [generate_report()](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/groq_report.py#20-26) with live Groq API
- [ ] Proper error handling (show errors on frontend if API fails)

### 3. Rewrite [backend/database.py](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/database.py) – Full MySQL Schema
- [ ] Proper sessions table with all fields (score, severity, etc.)
- [ ] Sensor readings table for time-series data
- [ ] Wire database into app.py routes

### 4. Rewrite [backend/app.py](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/app.py) – Real Camera Analysis
- [ ] Integrate MediaPipe FaceMesh + Pose for server-side analysis
- [ ] Real-time camera frame processing via `/api/camera_frame`
- [ ] Eye-blink detection (EAR algorithm)
- [ ] Facial expression estimation
- [ ] Body posture scoring
- [ ] Replace mock sensor data with real camera analysis results
- [ ] ESP32 serial integration (with graceful fallback)
- [ ] Wire up database for session storage and history
- [ ] Wire up Groq for adaptive questions and report generation
- [ ] Proper fallback logic (camera-only / sensor-only / questions-only)
- [ ] Error indicators on frontend for failed components

### 5. Update [backend/models.py](file:///Users/eashanjain/Documents/Minor-eeg-project/backend/models.py) – Data Classes
- [ ] Full session model matching database schema
- [ ] Sensor reading model

### 6. Update Frontend Integration
- [ ] Update [dashboard.js](file:///Users/eashanjain/Documents/Minor-eeg-project/frontend/static/js/dashboard.js) to send camera frames to backend
- [ ] Draw MediaPipe landmarks on camera overlay canvas
- [ ] Add test instructions panel
- [ ] Show real error messages for failed components
- [ ] Visual camera enhancements (mirror, brightness)

### 7. Update [dashboard.html](file:///Users/eashanjain/Documents/Minor-eeg-project/frontend/templates/dashboard.html) – Instructions Panel
- [ ] Add test instructions for efficient testing

## Verification
- [ ] Run Flask server and test dashboard
- [ ] Test camera analysis with real webcam
- [ ] Test Groq API integration
- [ ] Test fallback when camera/sensors unavailable
- [ ] Test error handling when Groq API fails
