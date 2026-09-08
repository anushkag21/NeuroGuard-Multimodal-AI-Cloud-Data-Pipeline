/*
  ============================================================
  NeuroGuard Clinic - Final Sensor Hub (EMG + MAX30102)
  ============================================================

  ESP32 Wiring:

  MAX30102:
    3V3 -> VIN
    GND -> GND
    GPIO21 -> SDA
    GPIO22 -> SCL

  EMG Sensor:
    OUT -> GPIO34
    VCC -> 3.3V
    GND -> GND

  Output:
  DATA:IR=...,RED=...,HR=...,SPO2=...,EMG=...,MUSCLE=...,BEAT=...,TS=...
  ============================================================
*/

#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

// ── EMG ─────────────────────────────────────────────────────
const int emgPin = 34;
int emgRaw = 0;
int emgProcessed = 0;

float emgFiltered = 0;
const float alpha = 0.2;

const int EMG_THRESHOLD = 1500;
int muscleActive = 0;

// ── Beat Detection ──────────────────────────────────────────
bool detectBeat(long irValue) {
  static long s0 = 0, s1 = 0, s2 = 0;
  static long lastBeatMs = 0;
  static long irMin = 999999, irMax = 0;

  s0 = s1;
  s1 = s2;
  s2 = irValue;

  if (irValue < irMin) irMin = irValue;
  else irMin += (irValue - irMin) / 2000;

  if (irValue > irMax) irMax = irValue;
  else irMax -= (irMax - irValue) / 2000;

  long amplitude = irMax - irMin;
  if (amplitude < 300) return false;

  bool isPeak = (s1 > s0 && s1 > s2);
  long midpoint = (irMin + irMax) / 2;
  bool aboveThreshold = (s1 > midpoint);
  bool refractoryOk = (millis() - lastBeatMs > 260);

  if (isPeak && aboveThreshold && refractoryOk) {
    lastBeatMs = millis();
    return true;
  }
  return false;
}

// ── Timing ──────────────────────────────────────────────────
unsigned long lastReportTs = 0;
const int REPORT_INTERVAL_MS = 100;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("STATUS:INIT_START");

  Wire.begin(21, 22);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("STATUS:MAX30102_ERROR");
    while (1);
  }

  particleSensor.setup(60, 4, 2, 100, 411, 4096);
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeIR(0x1F);

  Serial.println("STATUS:MAX30102_OK");
}

void loop() {

  // ── Read Sensors ──────────────────────────────────────────
  long irValue  = particleSensor.getIR();
  long redValue = particleSensor.getRed();

  emgRaw = analogRead(emgPin);

  // EMG Processing (smoothing)
  emgFiltered = alpha * emgRaw + (1 - alpha) * emgFiltered;
  emgProcessed = (int)emgFiltered;

  // Muscle detection
  muscleActive = (emgProcessed > EMG_THRESHOLD) ? 1 : 0;

  bool fingerDetected = (irValue > 50000);

  // ── Heart Rate ────────────────────────────────────────────
  static float bpmBuffer[6] = {0};
  static int bpmIndex = 0;
  static float beatsPerMinute = 0;
  static long lastBeatMs = 0;
  static bool firstBeat = true;
  int beatDetected = 0;

  if (fingerDetected && detectBeat(irValue)) {
    long now = millis();
    long delta = now - lastBeatMs;
    lastBeatMs = now;

    if (firstBeat) {
      firstBeat = false;
    } 
    else if (delta > 250 && delta < 2000) {
      float bpm = 60000.0 / delta;
      bpmBuffer[bpmIndex % 6] = bpm;
      bpmIndex++;

      float sum = 0;
      int n = min(bpmIndex, 6);
      for (int i = 0; i < n; i++) sum += bpmBuffer[i];

      beatsPerMinute = sum / n;
      beatDetected = 1;
    }
  }

  if (!fingerDetected) {
    beatsPerMinute = 0;
    bpmIndex = 0;
    firstBeat = true;
    lastBeatMs = 0;
  }

  // ── SpO2 ─────────────────────────────────────────────────
  static int spo2 = 0;
  static double avered = 0, aveir = 0;
  static double sumredrms = 0, sumirrms = 0;
  static int count = 0;

  if (fingerDetected) {
    double fred = redValue;
    double fir  = irValue;

    avered = avered * 0.95 + fred * 0.05;
    aveir  = aveir  * 0.95 + fir  * 0.05;

    sumredrms += (fred - avered) * (fred - avered);
    sumirrms  += (fir  - aveir)  * (fir  - aveir);
    count++;

    if (count >= 25) {
      double R = (sqrt(sumredrms) / avered) / (sqrt(sumirrms) / aveir);
      int newSpo2 = (int)(-45.060 * R * R + 30.354 * R + 94.845);
      spo2 = constrain(newSpo2, 70, 100);

      sumredrms = sumirrms = 0;
      count = 0;
    }
  } else {
    spo2 = 0;
    avered = aveir = sumredrms = sumirrms = 0;
    count = 0;
  }

  // ── Output ───────────────────────────────────────────────
  if (millis() - lastReportTs >= REPORT_INTERVAL_MS) {
    lastReportTs = millis();

    int hrOut   = fingerDetected ? (int)beatsPerMinute : 0;
    int spo2Out = fingerDetected ? spo2 : 0;

    Serial.print("DATA:");
    Serial.print("IR=");    Serial.print(irValue);
    Serial.print(",RED=");  Serial.print(redValue);
    Serial.print(",HR=");   Serial.print(hrOut);
    Serial.print(",SPO2="); Serial.print(spo2Out);
    Serial.print(",EMG=");  Serial.print(emgProcessed);
    Serial.print(",MUSCLE="); Serial.print(muscleActive);
    Serial.print(",BEAT="); Serial.print(beatDetected);
    Serial.print(",TS=");   Serial.print(millis());
    Serial.println();
  }

  delay(20);
}