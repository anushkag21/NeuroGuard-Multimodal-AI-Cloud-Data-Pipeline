/*
  ============================================================
  NeuroGuard Clinic - MAX30102 Sensor Hub (Final Clean)
  ============================================================

  Wiring (ESP32 DevKit v1 -> MAX30102):
    3V3 -> VIN  (DO NOT USE 5V)
    GND -> GND
    D21 -> SDA
    D22 -> SCL

  Output format (parsed by Python backend):
    DATA:IR=100340,RED=99420,HR=75,SPO2=98,BEAT=1,TS=45021
  ============================================================
*/

#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

// ── Beat Detector (raw IR, adaptive min/max) ──────────────────────────────────
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

  bool isPeak         = (s1 > s0 && s1 > s2);
  long midpoint       = (irMin + irMax) / 2;
  bool aboveThreshold = (s1 > midpoint);
  bool refractoryOk   = (millis() - lastBeatMs > 260);

  if (isPeak && aboveThreshold && refractoryOk) {
    lastBeatMs = millis();
    return true;
  }
  return false;
}

// ── Report Interval ───────────────────────────────────────────────────────────
unsigned long lastReportTs       = 0;
const int     REPORT_INTERVAL_MS = 100;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("STATUS:STARTING_INIT");

  Wire.begin(21, 22);
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("STATUS:MAX30102_ERROR - Check Wiring or Power!");
    while (1);
  }

  particleSensor.setup(60, 4, 2, 100, 411, 4096);
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeIR(0x1F);

  Serial.println("STATUS:MAX30102_OK");
}

void loop() {
  long irValue  = particleSensor.getIR();
  long redValue = particleSensor.getRed();
  bool fingerDetected = (irValue > 50000);

  // ── Heart Rate ──────────────────────────────────────────────────────────────
  static float beatsPerMinute = 0;
  static float bpmBuffer[6]   = {0};
  static int   bpmIndex       = 0;
  static long  lastBeatMs     = 0;
  static bool  firstBeat      = true;
  int beatDetected = 0;

  if (fingerDetected && detectBeat(irValue)) {
    long now   = millis();
    long delta = now - lastBeatMs;
    lastBeatMs = now;

    if (firstBeat) {
      firstBeat = false;
    } else if (delta > 250 && delta < 2000) {
      float instantBPM            = 60000.0f / (float)delta;
      bpmBuffer[bpmIndex % 6]     = instantBPM;
      bpmIndex++;

      float sum = 0;
      int   n   = min(bpmIndex, 6);
      for (int i = 0; i < n; i++) sum += bpmBuffer[i];
      beatsPerMinute = sum / n;
      beatDetected   = 1;
    }
  }

  if (!fingerDetected) {
    beatsPerMinute = 0;
    bpmIndex       = 0;
    firstBeat      = true;
    lastBeatMs     = 0;
    for (int i = 0; i < 6; i++) bpmBuffer[i] = 0;
  }

  // ── SpO2 ────────────────────────────────────────────────────────────────────
  static int    spo2      = 0;
  static double avered    = 0, aveir = 0;
  static double sumredrms = 0, sumirrms = 0;
  static int    count     = 0;

  if (fingerDetected) {
    double fred = (double)redValue;
    double fir  = (double)irValue;

    avered = avered * 0.95 + fred * 0.05;
    aveir  = aveir  * 0.95 + fir  * 0.05;

    sumredrms += (fred - avered) * (fred - avered);
    sumirrms  += (fir  - aveir)  * (fir  - aveir);
    count++;

    if (count >= 25) {
      double R    = (sqrt(sumredrms) / avered) / (sqrt(sumirrms) / aveir);
      int newSpo2 = (int)(-45.060 * R * R + 30.354 * R + 94.845);
      spo2        = constrain(newSpo2, 70, 100);
      sumredrms   = sumirrms = 0;
      count       = 0;
    }
  } else {
    spo2      = 0;
    avered    = aveir = sumredrms = sumirrms = 0;
    count     = 0;
  }

  // ── Send to Python Backend at 10 FPS ─────────────────────────────────────
  if (millis() - lastReportTs >= REPORT_INTERVAL_MS) {
    lastReportTs = millis();

    int hrOut   = fingerDetected ? (int)beatsPerMinute : 0;
    int spo2Out = fingerDetected ? spo2 : 0;

    Serial.print("DATA:");
    Serial.print("IR=");    Serial.print(irValue);
    Serial.print(",RED=");  Serial.print(redValue);
    Serial.print(",HR=");   Serial.print(hrOut);
    Serial.print(",SPO2="); Serial.print(spo2Out);
    Serial.print(",BEAT="); Serial.print(beatDetected);
    Serial.print(",TS=");   Serial.print(millis());
    Serial.println();
  }

  delay(50);
}