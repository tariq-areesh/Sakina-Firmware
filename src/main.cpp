#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>

#include "model_data.h"
#include "types.h"
#include "Model.h"
#include "SensorManager.h"
#include "BLELink.h"
#include "Display.h"

// ---------------------------------------------------------------
// Global instances
// ---------------------------------------------------------------
static Model         model;
static SensorManager sensors;
static BLELink       ble;
static Display       display;

// ---------------------------------------------------------------
// Inference results  (updated each inference tick)
// ---------------------------------------------------------------
static float latestScore  = 0.0f;
static bool  latestStress = false;

// ---------------------------------------------------------------
// Timing
// ---------------------------------------------------------------
constexpr unsigned long BVP_SAMPLE_MS   = 1000;
constexpr unsigned long TEMP_SAMPLE_MS  = 1000;
constexpr unsigned long INFERENCE_MS    = 1000;
constexpr unsigned long BLE_SEND_MS     = 500;
constexpr unsigned long SCREEN_SWITCH_MS = 2500;
constexpr unsigned long STRESS_DRIFT_MS = 5000;

static unsigned long lastBvpMs    = 0;
static unsigned long lastTempMs   = 0;
static unsigned long lastInferMs  = 0;
static unsigned long lastBleMs    = 0;
static unsigned long lastScreenMs = 0;

static bool showMain = true;

// ---------------------------------------------------------------
// setup()
// ---------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(1500);
    randomSeed(micros());

    display.begin();
    model.load(g_model);
    ble.begin();
    sensors.begin();

    const unsigned long t = millis();
    lastBvpMs = lastTempMs = lastInferMs = lastBleMs = lastScreenMs = t;
}

// ---------------------------------------------------------------
// loop()
// ---------------------------------------------------------------
void loop() {
    const unsigned long now = millis();

    // --- BLE RX command ---
    if (ble.hasCommand()) {
        sensors.handleCommand(ble.recvCommand());
    }

    // --- Drift latent stress level ---
    sensors.updateStress(now);

    // --- BVP at 1 Hz ---
    if (now - lastBvpMs >= BVP_SAMPLE_MS) {
        lastBvpMs = now;
        sensors.pushBvpSample(now);
    }

    // --- TEMP at 1 Hz ---
    if (now - lastTempMs >= TEMP_SAMPLE_MS) {
        lastTempMs = now;
        sensors.pushTempSample(now);
    }

    // --- Inference at 1 Hz (once buffers are full) ---
    if (sensors.buffersReady() && now - lastInferMs >= INFERENCE_MS) {
        lastInferMs = now;
        Window w    = sensors.nextWindow();
        latestScore = model.predict(w);
        if (isnan(latestScore)) latestScore = 0.0f;
        latestStress = (latestScore >= Model::STRESS_THRESHOLD);
        Serial.print("Score: "); Serial.print(latestScore, 5);
        Serial.print("  Class: "); Serial.println(latestStress ? "STRESSED" : "NOT STRESSED");
    }

    // --- BLE TX at 2 Hz ---
    if (now - lastBleMs >= BLE_SEND_MS) {
        lastBleMs = now;
        ble.sendStatus(
            latestStress ? 1 : 0,
            latestScore,
            (uint32_t)now,
            sensors.getRawTempC(),
            sensors.getHr(),
            sensors.getSmoothedBvp(),
            Display::adviceText(latestStress, latestScore)
        );
    }

    // --- OLED: toggle screens every 2.5 s ---
    if (now - lastScreenMs >= SCREEN_SWITCH_MS) {
        lastScreenMs = now;
        showMain = !showMain;
    }
    if (showMain) {
        display.drawMain(sensors.getHr(), sensors.getRawTempC(),
                         latestScore, latestStress);
    } else {
        display.drawFeatures(sensors.featBvpMean, sensors.featBvpStd,
                             sensors.featTempMean, sensors.featTempStd,
                             sensors.featTempSlope,
                             sensors.getStressLevel(),
                             latestScore, latestStress);
    }

    // --- BLE reconnect ---
    ble.poll();

    delay(5);
}
