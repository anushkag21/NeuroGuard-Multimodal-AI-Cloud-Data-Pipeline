"""
NeuroGuard Clinic – Groq LLM Report Generator
Real Groq API integration for adaptive questions and report generation.
"""
import os
import json
import traceback
from dotenv import load_dotenv

load_dotenv()


class GroqReportGenerator:
    """Handles all Groq LLM interactions for NeuroGuard Clinic."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY', '')
        self.client = None
        self.error = None
        self._init_client()

    def _init_client(self):
        """Initialize the Groq client."""
        if not self.api_key:
            self.error = "GROQ_API_KEY not set in .env file"
            print(f"[GroqReport] WARNING: {self.error}")
            return
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            print("[GroqReport] Groq client initialized successfully")
        except Exception as e:
            self.error = f"Failed to initialize Groq client: {str(e)}"
            print(f"[GroqReport] ERROR: {self.error}")

    def is_available(self):
        """Check if Groq API is ready."""
        return self.client is not None and self.error is None

    def generate_adaptive_question(self, sensor_data, previous_answers, question_index):
        """
        Generate an AI-personalized adaptive question based on live sensor data.
        Returns dict with question text and options, or error dict.
        """
        if not self.is_available():
            return {'error': self.error or 'Groq API not available'}

        # Build sensor context
        camera = sensor_data.get('camera', {})
        blink_rate = camera.get('blink_rate', 'N/A')
        expressions = camera.get('facial_expression', {})
        sadness = expressions.get('sad', 'N/A')
        posture = camera.get('posture_score', 'N/A')

        eeg = sensor_data.get('eeg', {})
        theta = eeg.get('theta', 'N/A')
        hrv = sensor_data.get('hrv', {})
        hrv_bpm = hrv.get('bpm', 'N/A')

        answers_text = json.dumps(previous_answers) if previous_answers else "No answers yet"

        prompt = f"""You are an expert clinical psychologist specializing in depression screening.
You are using the PHQ-9 assessment framework enhanced with real-time multimodal sensor data from a clinic system.

Current patient's LIVE sensor readings:
- Eye blink rate: {blink_rate}/min (normal: 15-20, elevated may indicate fatigue/stress)
- Facial sadness score: {sadness}% (detected via computer vision)
- Body posture score: {posture}% (100=upright, lower=slumped)
- EEG theta power: {theta} (elevated theta may indicate drowsiness/rumination)
- Heart rate: {hrv_bpm} BPM
- EMG muscle tension: {sensor_data.get('emg', {}).get('tension_pct', 'N/A')}% active (elevated tension may indicate anxiety/stress)
- EMG raw value: {sensor_data.get('emg', {}).get('raw', 'N/A')} (threshold: 1500)

Previous answers given by patient: {answers_text}
This is adaptive question #{question_index} (total questions should stay under 9).

Based on the sensor data patterns and previous answers, generate EXACTLY 1 personalized follow-up question
that will improve depression detection accuracy. The question should be:
- Relevant to any anomalies in the sensor data (e.g., high blink rate → ask about sleep/fatigue, high EMG → ask about physical tension/anxiety)
- Different from previously asked questions
- Clinically relevant for depression assessment

Respond in ONLY this exact JSON format, nothing else:
{{"question": "Your question text here?", "reasoning": "Brief explanation of why this question was chosen based on sensor data"}}"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=300,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            return {
                'question': result.get('question', ''),
                'reasoning': result.get('reasoning', ''),
                'source': 'AI-Generated',
            }
        except json.JSONDecodeError:
            return {'error': 'Groq returned invalid JSON', 'raw': content if 'content' in dir() else ''}
        except Exception as e:
            return {'error': f'Groq API error: {str(e)}'}

    def analyze_question_response(self, question_text, answer_value, answer_label, sensor_snapshot):
        """
        Generate per-question AI analysis of the patient's response combined with sensor data.
        Returns a brief insight string.
        """
        if not self.is_available():
            return "AI analysis unavailable – Groq API not connected"

        camera = sensor_snapshot.get('camera', {})
        expressions = camera.get('facial_expression', {})

        prompt = f"""You are a clinical psychologist AI assistant analyzing a depression screening response.

Question asked: "{question_text}"
Patient's answer: "{answer_label}" (score: {answer_value}/3)

Sensor data at time of answer:
- Facial expression: neutral={expressions.get('neutral', 'N/A')}%, sad={expressions.get('sad', 'N/A')}%, happy={expressions.get('happy', 'N/A')}%
- Blink rate: {camera.get('blink_rate', 'N/A')}/min
- Posture: {camera.get('posture_score', 'N/A')}%
- EMG muscle tension: {sensor_snapshot.get('serial', {}).get('emg', {}).get('raw', 'N/A')} (active={sensor_snapshot.get('serial', {}).get('emg', {}).get('muscle_active', 'N/A')})

In exactly 1-2 sentences, provide a brief clinical observation about the congruence between the patient's
verbal response and their physiological/behavioral signals. Note any discrepancies.
Respond with ONLY the observation text, no JSON or formatting."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_completion_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Analysis unavailable: {str(e)}"

    def generate_report(self, score, severity, confidence, sensor_summary, answers_with_analysis, mode, duration):
        """
        Generate a comprehensive, structured depression analysis report.
        Returns dict with structured sections, or error dict.
        """
        if not self.is_available():
            return {'error': self.error or 'Groq API not available'}

        answers_text = ""
        for i, a in enumerate(answers_with_analysis, 1):
            answers_text += f"Q{i}: \"{a.get('question', '')}\" → \"{a.get('answer_label', '')}\" (score {a.get('answer_value', 0)})\n"
            if a.get('ai_analysis'):
                answers_text += f"   AI Observation: {a['ai_analysis']}\n"

        prompt = f"""You are a senior clinical psychologist AI generating a professional depression screening report.

SESSION DATA:
- Final PHQ-9 equivalent score: {score}/27
- Severity classification: {severity}
- AI confidence: {confidence}%
- Assessment mode: {mode}
- Test duration: {duration}

SENSOR SUMMARY:
{json.dumps(sensor_summary, indent=2)}

QUESTION RESPONSES WITH AI OBSERVATIONS:
{answers_text}

Generate a structured clinical report in this EXACT JSON format:
{{
  "executive_summary": "2-3 sentence overview of findings",
  "key_observations": [
    "Observation 1 about physiological markers",
    "Observation 2 about behavioral patterns",
    "Observation 3 about response patterns",
    "Observation 4 about EMG/muscle tension if data available"
  ],
  "sensor_analysis": {{
    "facial_behavioral": "Analysis of facial expressions and eye patterns",
    "physiological": "Analysis of posture and any available HRV data",
    "emg_analysis": "Analysis of EMG muscle tension patterns — elevated tension may correlate with anxiety, restlessness, or psychomotor agitation. Low/absent tension in context of depression may suggest psychomotor retardation.",
    "response_congruence": "How well verbal responses match physiological data"
  }},
  "risk_assessment": "Assessment of risk level with clinical reasoning",
  "recommendations": [
    "Specific recommendation 1",
    "Specific recommendation 2",
    "Specific recommendation 3",
    "Specific recommendation 4"
  ],
  "disclaimer": "Standard medical disclaimer"
}}

Be professional, empathetic, evidence-based. Reference specific sensor readings where relevant.
Do NOT make a medical diagnosis – frame as screening results that should be discussed with a doctor."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_completion_tokens=1200,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            report = json.loads(content)
            return report
        except json.JSONDecodeError:
            return {'error': 'Groq returned invalid JSON for report'}
        except Exception as e:
            return {'error': f'Groq API error: {str(e)}'}
