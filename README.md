# NeuroGuard Clinic 🧠

### Multimodal AI System for Mental-Health State Analysis

NeuroGuard Clinic is a **multimodal AI and data-analysis project** designed to explore how physiological, behavioral, and self-reported signals can be combined to estimate a user's mental-health state.

Instead of relying on a single questionnaire or signal, the system brings together multiple data modalities such as:

* 🧠 **EEG** — brainwave frequency-band information
* ❤️ **HRV / Heart Rate** — physiological indicators
* 👁️ **Facial & behavioral cues** — camera-based analysis
* 🧍 **Posture / movement** — behavioral features
* 📝 **Questionnaire responses** — self-reported information

The project demonstrates an end-to-end pipeline from **data acquisition → preprocessing → feature extraction → multimodal fusion → visualization → report generation**.

> **Note:** NeuroGuard is an experimental research/prototype system and is **not a medical diagnostic tool**. Its purpose is to demonstrate multimodal data processing and AI-assisted analysis.

---

## 🎯 Problem Statement

Mental-health assessment often relies heavily on self-reported questionnaires. However, human mental states can manifest through multiple measurable signals.

NeuroGuard explores the following question:

> **Can heterogeneous physiological, behavioral, and questionnaire data be combined to obtain a richer representation of a user's mental-health state?**

The project therefore focuses on **multimodal data integration**, where different signals are processed independently and then combined to produce a unified analysis.

---

## 🔬 Data Science Pipeline

The core idea of the project is an end-to-end multimodal pipeline:

```text
             Raw Data
                 │
     ┌───────────┼────────────┐
     ↓           ↓            ↓
    EEG         HRV       Camera Data
     │           │            │
     ↓           ↓            ↓
Signal        HRV          Behavioral
Processing    Features      Features
     │           │            │
     └───────────┼────────────┘
                 ↓
        Feature Integration
                 ↓
        Multimodal Analysis
                 ↓
       Combined Mental-State
             Indicators
                 ↓
       Visualization / Report
```

### Key Data Science Components

**1. Signal Processing**

Physiological signals require preprocessing before meaningful features can be extracted. The project is structured to support:

* Noise reduction
* Signal filtering
* Artifact handling
* Window-based analysis
* Frequency-domain analysis

**2. Feature Extraction**

Examples of features considered include:

* EEG frequency bands
* Heart-rate / HRV measures
* Facial-expression indicators
* Blink-related features
* Posture and movement landmarks
* Questionnaire-derived scores

**3. Multimodal Feature Integration**

The project explores combining heterogeneous features from different sources rather than treating each modality independently.

This provides the foundation for future **machine-learning based multimodal classification/regression models**.

---

## 🤖 AI / Machine Learning Direction

The architecture is designed around a multimodal learning approach in which individual modalities can be modeled separately and their representations combined.

Planned / extensible model architecture includes:

* **CNN / CNN-LSTM** for temporal physiological signals
* **Computer Vision models** for facial/behavioral features
* **Attention-based multimodal fusion**
* **Feature-level / late fusion**
* ML-based mental-state estimation

The current repository is primarily an **end-to-end prototype**, with the data-processing and model interfaces structured so that trained models and real sensor streams can be integrated as the next development stage.

---

## 📊 Multimodal Inputs

| Modality      | Example Information       | Processing                             |
| ------------- | ------------------------- | -------------------------------------- |
| EEG           | Frequency-band activity   | Signal processing / feature extraction |
| HRV           | Heart-rate variability    | Time-series analysis                   |
| Camera        | Facial / behavioral cues  | Computer vision                        |
| Posture       | Body landmarks / movement | Landmark analysis                      |
| Questionnaire | Self-reported responses   | Score-based features                   |

The important aspect of the project is the **combination of heterogeneous data sources** into a common analysis pipeline.

---

## 🏗️ System Architecture

```text
                    NeuroGuard Clinic
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    Physiological       Camera            Questionnaire
       Signals            Data                Data
        │                  │                  │
        ↓                  ↓                  ↓
      Python            Computer            Response
     Processing          Vision             Processing
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                  Feature Integration
                           ↓
                    AI / Analysis Layer
                           ↓
                 Flask Backend / API
                           ↓
                Interactive Dashboard
                           ↓
                   Analysis Report
```

---

## 🖥️ Application

The project includes an interactive web dashboard for presenting multimodal analysis.

### Dashboard

Provides a visual interface for:

* Physiological signal monitoring
* EEG feature visualization
* HRV information
* Camera-based indicators
* Questionnaire responses
* Combined analysis scores

### Reports

The system provides a structured report interface for presenting the analysis generated from a session.

### Session History

Previous sessions can be stored and reviewed to enable comparison across multiple assessments.

---

## 🛠️ Tech Stack

### Data Science / AI

* Python
* NumPy
* SciPy
* TensorFlow
* OpenCV
* MediaPipe

### Backend

* Flask
* Python
* MySQL

### Frontend

* HTML
* CSS
* Bootstrap
* JavaScript
* Chart.js
* Jinja2

### Hardware

* ESP32
* BioAmp EXG Pill
* MAX30105
* Camera

### AI-Assisted Generation

* Groq API for experimental adaptive-question/report generation

---

## 📁 Project Structure

```text
NeuroGuard/
│
├── backend/
│   ├── app.py
│   ├── database.py
│   ├── models.py
│   └── groq_report.py
│
├── python_core/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── real_time_neuroguard_mac.py
│
├── hardware/
│   └── esp32_sensor_hub.ino
│
├── frontend/
│   ├── templates/
│   └── static/
│
├── requirements.txt
└── README.md
```

---

## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd NeuroGuard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Flask application

```bash
cd backend
python app.py
```

### 4. Open the dashboard

```text
http://127.0.0.1:5000
```

---

## 🔮 Future Work

The project is designed to be extended into a complete multimodal ML pipeline.

Planned improvements include:

* Real EEG sensor integration
* Real-time physiological signal acquisition
* Automated EEG preprocessing and artifact removal
* Robust HRV feature extraction
* Automated facial-expression feature extraction
* Automated posture feature extraction
* Dataset collection and labeling
* Training and validation of ML models
* CNN-LSTM based temporal modeling
* Attention-based multimodal fusion
* Model evaluation using appropriate statistical and ML metrics
* Personalized longitudinal analysis

---

## 📌 Why This Project?

NeuroGuard combines multiple areas of data science and AI in one system:

**Signal Processing + Time-Series Analysis + Computer Vision + Feature Engineering + Multimodal Learning + Data Visualization**

The project was built to explore how different types of real-world data can be processed independently and integrated into a single analytical framework.

---

## ⚠️ Disclaimer

NeuroGuard Clinic is an **academic/experimental prototype** intended to demonstrate data processing, AI, and multimodal analysis concepts.

It does **not provide a clinical diagnosis**, and its outputs should not be interpreted as medical advice.
