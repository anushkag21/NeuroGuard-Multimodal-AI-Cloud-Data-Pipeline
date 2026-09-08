#include <Wire.h>
// #include <MAX30105.h>  // Uncomment when MAX30105 library is installed
// MAX30105 particleSensor;

// NeuroGuard Clinic – ESP32 Sensor Hub (Stub)
// Upload this to your ESP32 after connecting BioAmp + MAX30105

void setup() {
    Serial.begin(115200);
    pinMode(34, INPUT);  // BioAmp EXG Pill on GPIO34 (ADC1)

    // Uncomment when MAX30105 is connected:
    // if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    //     Serial.println("MAX30105 not found");
    // }
    // particleSensor.setup();

    Serial.println("NeuroGuard ESP32 Hub Ready");
}

void loop() {
    int bioampValue = analogRead(34);  // EEG/EMG raw
    // float hr = particleSensor.getRed();  // Uncomment with sensor

    Serial.print("BIOAMP:");
    Serial.print(bioampValue);
    Serial.print(",HR:");
    Serial.println(0);  // Replace 0 with hr when sensor connected
    delay(20);
}
