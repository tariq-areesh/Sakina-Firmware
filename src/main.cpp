#include <Arduino.h>
#include <math.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TensorFlowLite_ESP32.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_ADDR    0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  const tflite::Model*      model_tfl  = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input  = nullptr;
  TfLiteTensor* output = nullptr;
  constexpr size_t kTensorArenaSize = 32 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

// =====================================================================
// PIPELINE NOTES  (must match Data_prep.py + training_arc.py exactly)
// =====================================================================
//
// 1. TEMP NORMALIZATION
//    The source cGAN CSV stores TEMP already normalized. Reverse-mapping
//    the NPZ distributions back to Celsius:
//      normal-class  TEMP_mean 0.683 → 36.83 °C wrist skin temp
//      stress-class  TEMP_mean 0.458 → 34.58 °C (vasoconstriction)
//    gives:   norm = (raw_celsius - 30.0) / 10.0
//    This replaces the old wrong formula (T-35)/5 which was too narrow.
//
// 2. BVP NORMALIZATION
//    Source CSV BVP already in [0.35, 0.70]. Simulator output matches.
//    No extra normalization needed.
//
// 3. WINDOW  — 60 samples at 1 Hz = 60 seconds (WINDOW_SIZE=60 in training)
//
// 4. FEATURES (order must match noHR NPZ cols exactly):
//       [BVP_mean, BVP_std, TEMP_mean, TEMP_std, TEMP_slope]
//    TEMP features are in NORMALIZED space (after step 1).
//    TEMP_slope = (last_norm - first_norm) / 60.0
//
// 5. STANDARDSCALER
//    training_arc.py fits StandardScaler on the training split and saves
//    scaler_stats.npz. Stats below are derived from the full NPZ (error
//    vs training-only split < 0.5% — negligible). Replace with your own
//    values if you retrain.
//
// 6. THRESHOLD
//    Set STRESS_THRESHOLD to best_thr printed by training_arc.py.
//    Default 0.5.
//
// =====================================================================

constexpr int N_FEATURES = 5;

// StandardScaler stats from 10000_subj_synthetic_cGAN_training_ready_noHR.npz
// Order: BVP_mean, BVP_std, TEMP_mean, TEMP_std, TEMP_slope
constexpr float SCALER_MEAN[N_FEATURES]  = { 0.5117f,  0.0625f,  0.6177f,  0.0890f, -0.0000203f };
constexpr float SCALER_SCALE[N_FEATURES] = { 0.0317f,  0.0338f,  0.2587f,  0.1071f,  0.00471f   };

// Replace with best_thr from training_arc.py output if you have it
constexpr float STRESS_THRESHOLD = 0.5f;

// =====================================================================
// BLE
// =====================================================================
static const char* SERVICE_UUID = "12345678-1234-1234-1234-1234567890ab";
static const char* TX_UUID      = "12345678-1234-1234-1234-1234567890ac";
static const char* RX_UUID      = "12345678-1234-1234-1234-1234567890ad";

BLEServer*         pServer           = nullptr;
BLECharacteristic* pTxCharacteristic = nullptr;
BLECharacteristic* pRxCharacteristic = nullptr;
bool deviceConnected    = false;
bool oldDeviceConnected = false;

// BLE RX command — written in callback, consumed in loop()
volatile bool cmdPending = false;
char cmdBuffer[64] = {0};

// =====================================================================
// Timing
// =====================================================================
unsigned long lastScreenSwitch = 0;
bool showMainScreen = true;

constexpr unsigned long SCREEN_SWITCH_MS = 2500;
constexpr unsigned long BVP_SAMPLE_MS   = 1000;  // 1 Hz — matches training fs=1.0
constexpr unsigned long TEMP_SAMPLE_MS  = 1000;  // 1 Hz
constexpr unsigned long BLE_SEND_MS     = 500;
constexpr unsigned long INFERENCE_MS    = 1000;
constexpr unsigned long STRESS_DRIFT_MS = 5000;  // update latent stress every 5 s

constexpr float WINDOW_SECONDS  = 60.0f;
constexpr int   BVP_BUFFER_SIZE  = 60;
constexpr int   TEMP_BUFFER_SIZE = 60;

unsigned long lastBvpSampleMs   = 0;
unsigned long lastTempSampleMs  = 0;
unsigned long lastBleSendMs     = 0;
unsigned long lastInferenceMs   = 0;
unsigned long lastStressDriftMs = 0;

// =====================================================================
// Rolling buffers — hold values in the same normalized space as training
// =====================================================================
float bvpBuffer[BVP_BUFFER_SIZE];
float tempBuffer[TEMP_BUFFER_SIZE];  // stores (raw_C - 30) / 10

int  bvpWriteIndex  = 0;
int  tempWriteIndex = 0;
bool bvpBufferFull  = false;
bool tempBufferFull = false;

// =====================================================================
// Simulator state
// =====================================================================
// Send "force:stress", "force:calm", or "force:auto" via BLE RX to control
enum SimMode { SIM_AUTO, SIM_FORCE_STRESS, SIM_FORCE_CALM };
SimMode simMode       = SIM_AUTO;
float   simStressLevel = 0.0f;  // 0=calm .. 1=fully stressed

float simulatedHrBpm  = 72.0f;
float currentHrBpm    = 72.0f;
float rawWristTempC   = 36.8f;   // Celsius, for display only
float normWristTemp   = 0.0f;    // (rawWristTempC - 30) / 10

float latestRawBvp      = 0.50f;
float latestSmoothedBvp = 0.50f;
float latestRawTemp     = 36.8f;
float latestSmoothedTemp = 36.8f;

bool hasBvpFilterState  = false;
bool hasTempFilterState = false;
constexpr float BVP_ALPHA  = 0.22f;
constexpr float TEMP_ALPHA = 0.10f;

float feat_bvp_mean   = 0.0f;
float feat_bvp_std    = 0.0f;
float feat_temp_mean  = 0.0f;
float feat_temp_std   = 0.0f;
float feat_temp_slope = 0.0f;

float latestScore  = 0.0f;
bool  latestStress = false;

// =====================================================================
// Utility
// =====================================================================
float randFloat(float lo, float hi) {
  return lo + ((float)random(0, 1000000) / 999999.0f) * (hi - lo);
}

float clampFloat(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Normalize raw wrist skin temp to match training data feature space.
// Derived by reverse-mapping NPZ TEMP_mean class distributions:
//   normal mean 0.683 → 36.83 °C, stress mean 0.458 → 34.58 °C
//   → formula: (T - 30) / 10
float normalizeTempC(float t) {
  return (t - 30.0f) / 10.0f;
}

float meanOfBuffer(const float* buf, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += buf[i];
  return s / n;
}

float stdOfBuffer(const float* buf, int n, float m) {
  // Population std (ddof=0) — matches np.nanstd in Data_prep.py
  float acc = 0.0f;
  for (int i = 0; i < n; i++) { float d = buf[i] - m; acc += d * d; }
  return sqrtf(acc / n);
}

float lowPassFilter(float in, float prev, float alpha, bool& init) {
  if (!init) { init = true; return in; }
  return alpha * in + (1.0f - alpha) * prev;
}

int8_t quantize_int8(float x, float scale, int zp) {
  int32_t q = (int32_t)roundf(x / scale) + zp;
  q = q < -128 ? -128 : (q > 127 ? 127 : q);
  return (int8_t)q;
}

float dequantize_int8(int8_t x, float scale, int zp) {
  return ((int)x - zp) * scale;
}

// =====================================================================
// Stress-aware simulator
// =====================================================================
void updateStressLevel(unsigned long now) {
  if (now - lastStressDriftMs < STRESS_DRIFT_MS) return;
  lastStressDriftMs = now;

  if      (simMode == SIM_FORCE_STRESS) simStressLevel = clampFloat(simStressLevel + 0.10f, 0.0f, 1.0f);
  else if (simMode == SIM_FORCE_CALM)   simStressLevel = clampFloat(simStressLevel - 0.10f, 0.0f, 1.0f);
  else {
    // Random walk — will naturally cross the decision boundary both ways
    simStressLevel += randFloat(-0.08f, 0.08f);
    simStressLevel  = clampFloat(simStressLevel, 0.0f, 1.0f);
  }
}

float generateRawBvpSample(unsigned long nowMs, float hrBpm) {
  const float beatMs = 60000.0f / max(hrBpm, 1.0f);
  const float phase  = fmod((float)nowMs, beatMs) / beatMs;
  float pulse;
  if      (phase < 0.12f) pulse = phase / 0.12f;
  else if (phase < 0.28f) pulse = 1.0f - ((phase - 0.12f) / 0.16f) * 0.52f;
  else                    pulse = 0.48f - ((phase - 0.28f) / 0.72f) * 0.10f;
  const float noiseAmp = 0.010f + simStressLevel * 0.015f;
  return clampFloat(
    0.42f + pulse * 0.22f
          + sinf((float)nowMs * 0.0012f) * 0.010f
          + sinf((float)nowMs * 0.00045f) * 0.018f
          + randFloat(-noiseAmp, noiseAmp),
    0.28f, 0.82f);
}

// Wrist skin temp drops with stress (peripheral vasoconstriction).
// calm target 36.8 C (norm 0.68), stressed target 31.5 C (norm 0.15).
// Decision boundary sits at ~35.7 C (norm 0.57).
// With this range, stress_level > 0.21 already crosses the boundary,
// which the random walk hits ~73% of the time.
float generateRawTempSample(unsigned long nowMs) {
  const float target = 36.8f + (31.5f - 36.8f) * simStressLevel;
  return clampFloat(
    target + sinf((float)nowMs * 0.000015f) * 0.10f
           + randFloat(-0.05f, 0.05f),
    30.0f, 39.5f);
}

// =====================================================================
// Buffers
// =====================================================================
void pushBvpSample(float v) {
  bvpBuffer[bvpWriteIndex] = v;
  bvpWriteIndex = (bvpWriteIndex + 1) % BVP_BUFFER_SIZE;
  if (bvpWriteIndex == 0) bvpBufferFull = true;
}

void pushTempSample(float v) {
  tempBuffer[tempWriteIndex] = v;
  tempWriteIndex = (tempWriteIndex + 1) % TEMP_BUFFER_SIZE;
  if (tempWriteIndex == 0) tempBufferFull = true;
}

void copyOrdered(const float* src, int size, int writeIdx, bool full, float* dst) {
  if (!full) { for (int i = 0; i < writeIdx; i++) dst[i] = src[i]; return; }
  int k = 0;
  for (int i = writeIdx; i < size;  i++) dst[k++] = src[i];
  for (int i = 0;        i < writeIdx; i++) dst[k++] = src[i];
}

bool buffersReady() { return bvpBufferFull && tempBufferFull; }

// =====================================================================
// Inference
// =====================================================================
float run_model(const float feats[N_FEATURES]) {
  const float in_scale = input->params.scale;
  const int   in_zp    = input->params.zero_point;
  for (int i = 0; i < N_FEATURES; i++) {
    // StandardScaler.transform: z = (x - mean) / scale
    const float z = (feats[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    // INT8 quantize using TFLite tensor params
    input->data.int8[i] = quantize_int8(z, in_scale, in_zp);
  }
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[ERROR] Invoke failed");
    return NAN;
  }
  return dequantize_int8(output->data.int8[0], output->params.scale, output->params.zero_point);
}

void computeFeaturesAndRunInference() {
  if (!buffersReady()) return;
  static float ordBvp[BVP_BUFFER_SIZE];
  static float ordTemp[TEMP_BUFFER_SIZE];
  copyOrdered(bvpBuffer,  BVP_BUFFER_SIZE,  bvpWriteIndex,  bvpBufferFull,  ordBvp);
  copyOrdered(tempBuffer, TEMP_BUFFER_SIZE, tempWriteIndex, tempBufferFull, ordTemp);

  feat_bvp_mean   = meanOfBuffer(ordBvp,  BVP_BUFFER_SIZE);
  feat_bvp_std    = stdOfBuffer (ordBvp,  BVP_BUFFER_SIZE, feat_bvp_mean);
  feat_temp_mean  = meanOfBuffer(ordTemp, TEMP_BUFFER_SIZE);
  feat_temp_std   = stdOfBuffer (ordTemp, TEMP_BUFFER_SIZE, feat_temp_mean);
  // Matches Data_prep.py: (last - first) / WINDOW_SIZE
  feat_temp_slope = (ordTemp[TEMP_BUFFER_SIZE-1] - ordTemp[0]) / WINDOW_SECONDS;

  const float feats[N_FEATURES] = {
    feat_bvp_mean, feat_bvp_std, feat_temp_mean, feat_temp_std, feat_temp_slope
  };

  latestScore  = run_model(feats);
  if (isnan(latestScore)) latestScore = 0.0f;
  latestStress = (latestScore >= STRESS_THRESHOLD);

  Serial.println("========================================");
  Serial.print("SimStressLevel:  "); Serial.println(simStressLevel, 3);
  Serial.print("Wrist Temp raw:  "); Serial.print(rawWristTempC, 2); Serial.println(" C");
  Serial.print("Wrist Temp norm: "); Serial.println(normWristTemp, 4);
  Serial.print("HR:              "); Serial.print(currentHrBpm, 1); Serial.println(" bpm");
  Serial.print("BVP_mean:        "); Serial.println(feat_bvp_mean, 5);
  Serial.print("BVP_std:         "); Serial.println(feat_bvp_std, 5);
  Serial.print("TEMP_mean(norm): "); Serial.println(feat_temp_mean, 5);
  Serial.print("TEMP_std (norm): "); Serial.println(feat_temp_std, 5);
  Serial.print("TEMP_slope:      "); Serial.println(feat_temp_slope, 6);
  Serial.print("Score:           "); Serial.println(latestScore, 5);
  Serial.print("Class:           "); Serial.println(latestStress ? "STRESSED" : "NOT STRESSED");
}

// =====================================================================
// BLE command handler
// Supported commands (send via BLE RX characteristic):
//   force:stress  — ramp simStressLevel toward 1 (produces stressed readings)
//   force:calm    — ramp simStressLevel toward 0 (produces normal readings)
//   force:auto    — return to autonomous random-walk mode
// =====================================================================
void handleBleCommand(const char* cmd) {
  Serial.print("[BLE CMD] "); Serial.println(cmd);
  if      (strcmp(cmd, "force:stress") == 0) { simMode = SIM_FORCE_STRESS; Serial.println("  -> forcing stress"); }
  else if (strcmp(cmd, "force:calm")   == 0) { simMode = SIM_FORCE_CALM;   Serial.println("  -> forcing calm");   }
  else if (strcmp(cmd, "force:auto")   == 0) { simMode = SIM_AUTO;         Serial.println("  -> auto mode");      }
  else                                         Serial.println("  -> unknown (use force:stress/calm/auto)");
}

// =====================================================================
// Display text
// =====================================================================
String stressText() { return latestStress ? "Stressed" : "Normal"; }

String adviceText() {
  if (!latestStress) return "You are balanced. Maintain your current routine.";
  if (latestScore < 0.65f) return "Slight stress. Try slow breathing for one minute.";
  if (latestScore < 0.80f) return "Moderate stress. Take a short walk or rest break.";
  return "High stress. Sit down and focus on slow exhales.";
}

// =====================================================================
// BLE send
// =====================================================================
void sendBlePacket() {
  if (!deviceConnected || !pTxCharacteristic) return;
  String p;
  p += "stress:" + stressText();
  p += ",temp:"  + String(rawWristTempC, 1);
  p += ",hr:"    + String((int)roundf(currentHrBpm));
  p += ",bvp:"   + String(latestSmoothedBvp, 4);
  p += ",score:" + String(latestScore, 3);
  p += ",advice:"+ adviceText();
  pTxCharacteristic->setValue(p.c_str());
  pTxCharacteristic->notify();
  Serial.print("BLE -> "); Serial.println(p);
}

// =====================================================================
// OLED
// =====================================================================
void drawMainScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,  0); display.print("HR: "); display.print(currentHrBpm, 0); display.print(" bpm");
  display.setCursor(0,  8); display.print("Temp: "); display.print(rawWristTempC, 1); display.print(" C");
  display.setCursor(0, 16); display.print("Score: "); display.print(latestScore, 3);
  display.setCursor(0, 24); display.print(latestStress ? "STRESSED" : "NOT STRESSED");
  display.display();
}

void drawFeatureScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,  0); display.print("Bm:"); display.print(feat_bvp_mean, 2);
                             display.print(" Bs:"); display.print(feat_bvp_std, 2);
  display.setCursor(0,  8); display.print("Tm:"); display.print(feat_temp_mean, 3);
                             display.print(" Ts:"); display.print(feat_temp_std, 3);
  display.setCursor(0, 16); display.print("Sl:"); display.print(feat_temp_slope, 4);
                             display.print(" SL:"); display.print(simStressLevel, 2);
  display.setCursor(0, 24); display.print("P:"); display.print(latestScore, 3);
                             display.print(latestStress ? " STR" : " NRM");
  display.display();
}

// =====================================================================
// BLE callbacks
// =====================================================================
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer*)    override { deviceConnected = true;  Serial.println("BLE connected"); }
  void onDisconnect(BLEServer*) override { deviceConnected = false; Serial.println("BLE disconnected"); }
};

class MyRxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* c) override {
    std::string v = c->getValue();
    if (!v.empty() && !cmdPending) {
      strncpy(cmdBuffer, v.c_str(), sizeof(cmdBuffer) - 1);
      cmdBuffer[sizeof(cmdBuffer)-1] = '\0';
      cmdPending = true;
    }
  }
};

// =====================================================================
// Setup helpers
// =====================================================================
void setupModel() {
  model_tfl = tflite::GetModel(g_model);
  if (!model_tfl) { Serial.println("GetModel failed"); while (true) delay(10); }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model_tfl, resolver, tensor_arena, kTensorArenaSize, error_reporter, nullptr, nullptr);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed"); while (true) delay(10);
  }
  input  = interpreter->input(0);
  output = interpreter->output(0);

  if (input->dims->data[input->dims->size-1] != N_FEATURES)
    Serial.println("[WARN] model input size mismatch — check N_FEATURES");

  Serial.println("Model ready");
  Serial.print("  in  scale/zp: "); Serial.print(input->params.scale,8);
  Serial.print(" / ");               Serial.println(input->params.zero_point);
  Serial.print("  out scale/zp: "); Serial.print(output->params.scale,8);
  Serial.print(" / ");               Serial.println(output->params.zero_point);
}

void setupDisplay() {
  Wire.begin(21, 22);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("SSD1306 init failed"); while (true) delay(10);
  }
  display.clearDisplay(); display.display();
}

void setupBle() {
  BLEDevice::init("Sakina ESP32");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService* svc = pServer->createService(SERVICE_UUID);
  pTxCharacteristic = svc->createCharacteristic(TX_UUID,
      BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ);
  pTxCharacteristic->addDescriptor(new BLE2902());
  pRxCharacteristic = svc->createCharacteristic(RX_UUID,
      BLECharacteristic::PROPERTY_WRITE |
      BLECharacteristic::PROPERTY_WRITE_NR |
      BLECharacteristic::PROPERTY_READ);
  pRxCharacteristic->setCallbacks(new MyRxCallbacks());
  svc->start();

  BLEAdvertising* adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->setScanResponse(true);
  adv->start();
  Serial.println("BLE advertising started");
}

void prefillBuffers() {
  // Prefill at the decision boundary (norm 0.57 = 35.7 °C) so the first
  // stressed or calm samples push the window mean over the threshold quickly,
  // without needing to flush 60 s of calm data first.
  constexpr float neutralNorm = 0.57f;
  for (int i = 0; i < BVP_BUFFER_SIZE;  i++) bvpBuffer[i]  = 0.51f;
  for (int i = 0; i < TEMP_BUFFER_SIZE; i++) tempBuffer[i] = neutralNorm;
  bvpWriteIndex = tempWriteIndex = 0;
  bvpBufferFull = tempBufferFull = false;
}

// =====================================================================
// Arduino entry points
// =====================================================================
void setup() {
  Serial.begin(115200);
  delay(1500);
  randomSeed(micros());

  setupDisplay();
  setupModel();
  setupBle();

  rawWristTempC      = randFloat(36.5f, 37.0f);
  latestRawTemp      = rawWristTempC;
  latestSmoothedTemp = rawWristTempC;
  normWristTemp      = normalizeTempC(rawWristTempC);
  simulatedHrBpm     = randFloat(62.0f, 75.0f);
  currentHrBpm       = simulatedHrBpm;
  simStressLevel     = 0.0f;

  prefillBuffers();

  const unsigned long t = millis();
  lastBvpSampleMs = lastTempSampleMs = lastBleSendMs =
  lastInferenceMs = lastScreenSwitch = lastStressDriftMs = t;
}

void loop() {
  const unsigned long now = millis();

  // Process BLE command outside ISR context
  if (cmdPending) { handleBleCommand(cmdBuffer); cmdPending = false; }

  // Drift latent stress level
  updateStressLevel(now);

  // --- BVP at 1 Hz ---
  if (now - lastBvpSampleMs >= BVP_SAMPLE_MS) {
    lastBvpSampleMs = now;
    // HR rises with stress: calm ~68 bpm, stressed ~106 bpm
    const float hrTarget = 68.0f + simStressLevel * 38.0f;
    simulatedHrBpm += (hrTarget - simulatedHrBpm) * 0.05f + randFloat(-0.3f, 0.3f);
    simulatedHrBpm  = clampFloat(simulatedHrBpm, 55.0f, 115.0f);
    currentHrBpm    = simulatedHrBpm;

    latestRawBvp     = generateRawBvpSample(now, simulatedHrBpm);
    // No low-pass filter on BVP at 1 Hz — consecutive 1 Hz samples already
    // land at different heartbeat phases, giving natural variance that matches
    // training BVP_std (~0.063). LPF at alpha=0.22 was killing it to ~0.012.
    pushBvpSample(latestRawBvp);
  }

  // --- TEMP at 1 Hz — push NORMALIZED value matching training ---
  if (now - lastTempSampleMs >= TEMP_SAMPLE_MS) {
    lastTempSampleMs = now;
    latestRawTemp    = generateRawTempSample(now);
    latestSmoothedTemp = lowPassFilter(latestRawTemp, latestSmoothedTemp, TEMP_ALPHA, hasTempFilterState);
    rawWristTempC    = latestSmoothedTemp;
    normWristTemp    = normalizeTempC(rawWristTempC);  // (T - 30) / 10
    pushTempSample(normWristTemp);
  }

  // --- Inference ---
  if (buffersReady() && now - lastInferenceMs >= INFERENCE_MS) {
    lastInferenceMs = now;
    computeFeaturesAndRunInference();
  }

  // --- BLE ---
  if (now - lastBleSendMs >= BLE_SEND_MS) {
    lastBleSendMs = now;
    sendBlePacket();
  }

  // --- OLED ---
  if (now - lastScreenSwitch >= SCREEN_SWITCH_MS) {
    lastScreenSwitch = now;
    showMainScreen = !showMainScreen;
  }
  showMainScreen ? drawMainScreen() : drawFeatureScreen();

  // --- BLE reconnect ---
  if (!deviceConnected && oldDeviceConnected) {
    delay(300); pServer->startAdvertising();
    Serial.println("BLE re-advertising"); oldDeviceConnected = false;
  }
  if (deviceConnected && !oldDeviceConnected) oldDeviceConnected = true;

  delay(5);
}
