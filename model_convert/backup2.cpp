#include <Arduino.h>
#include <math.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_ADDR 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr size_t kTensorArenaSize = 32 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

// Dataset-derived feature ranges
constexpr float BVP_MEAN_MIN   = 0.35249853f;
constexpr float BVP_MEAN_MAX   = 0.69602996f;

constexpr float BVP_STD_MIN    = 0.01302313f;
constexpr float BVP_STD_MAX    = 0.31201622f;

constexpr float TEMP_MEAN_MIN  = 0.0f;
constexpr float TEMP_MEAN_MAX  = 1.0646845f;

constexpr float TEMP_STD_MIN   = 0.0f;
constexpr float TEMP_STD_MAX   = 0.5369606f;

constexpr float TEMP_SLOPE_MIN = -0.01765528f;
constexpr float TEMP_SLOPE_MAX =  0.01779067f;

// Simulated live sensor values
float liveHR = 78.0f;
float liveBodyTemp = 36.8f;

// Latest inference inputs/results
float feat_bvp_mean   = 0.0f;
float feat_bvp_std    = 0.0f;
float feat_temp_mean  = 0.0f;
float feat_temp_std   = 0.0f;
float feat_temp_slope = 0.0f;
float latestScore     = 0.0f;
bool latestStress     = false;

// Timing
unsigned long lastSensorUpdate = 0;
unsigned long lastInference = 0;
unsigned long lastScreenSwitch = 0;
bool showMainScreen = true;

constexpr unsigned long SENSOR_UPDATE_MS = 250;
constexpr unsigned long INFERENCE_MS = 5000;
constexpr unsigned long SCREEN_SWITCH_MS = 2500;

float randFloat(float minVal, float maxVal) {
  long r = random(0, 1000000);
  float t = (float)r / 999999.0f;
  return minVal + t * (maxVal - minVal);
}

int8_t quantize_int8(float x, float scale, int zero_point) {
  int32_t q = (int32_t)roundf(x / scale) + zero_point;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (int8_t)q;
}

float dequantize_int8(int8_t x, float scale, int zero_point) {
  return ((int)x - zero_point) * scale;
}

void updateLiveVitals() {
  // Smooth fake HR variation
  liveHR += randFloat(-1.8f, 1.8f);
  if (liveHR < 62.0f) liveHR = 62.0f;
  if (liveHR > 108.0f) liveHR = 108.0f;

  // Smooth fake body temperature variation
  liveBodyTemp += randFloat(-0.05f, 0.05f);
  if (liveBodyTemp < 36.1f) liveBodyTemp = 36.1f;
  if (liveBodyTemp > 37.8f) liveBodyTemp = 37.8f;
}

float run_model(float f0, float f1, float f2, float f3, float f4) {
  float in_scale = input->params.scale;
  int in_zp = input->params.zero_point;

  input->data.int8[0] = quantize_int8(f0, in_scale, in_zp);
  input->data.int8[1] = quantize_int8(f1, in_scale, in_zp);
  input->data.int8[2] = quantize_int8(f2, in_scale, in_zp);
  input->data.int8[3] = quantize_int8(f3, in_scale, in_zp);
  input->data.int8[4] = quantize_int8(f4, in_scale, in_zp);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return NAN;
  }

  int8_t raw = output->data.int8[0];
  return dequantize_int8(raw, output->params.scale, output->params.zero_point);
}

void runRandomInference() {
  feat_bvp_mean   = randFloat(BVP_MEAN_MIN,   BVP_MEAN_MAX);
  feat_bvp_std    = randFloat(BVP_STD_MIN,    BVP_STD_MAX);
  feat_temp_mean  = randFloat(TEMP_MEAN_MIN,  TEMP_MEAN_MAX);
  feat_temp_std   = randFloat(TEMP_STD_MIN,   TEMP_STD_MAX);
  feat_temp_slope = randFloat(TEMP_SLOPE_MIN, TEMP_SLOPE_MAX);

  latestScore = run_model(
    feat_bvp_mean,
    feat_bvp_std,
    feat_temp_mean,
    feat_temp_std,
    feat_temp_slope
  );

  latestStress = (latestScore >= 0.5f);

  Serial.println("====================================");
  Serial.print("HR: "); Serial.println(liveHR, 1);
  Serial.print("Body Temp: "); Serial.println(liveBodyTemp, 2);
  Serial.print("BVP_mean: "); Serial.println(feat_bvp_mean, 6);
  Serial.print("BVP_std: "); Serial.println(feat_bvp_std, 6);
  Serial.print("TEMP_mean: "); Serial.println(feat_temp_mean, 6);
  Serial.print("TEMP_std: "); Serial.println(feat_temp_std, 6);
  Serial.print("TEMP_slope: "); Serial.println(feat_temp_slope, 6);
  Serial.print("Prediction: "); Serial.println(latestScore, 6);
  Serial.print("Class: "); Serial.println(latestStress ? "STRESSED" : "NOT STRESSED");
}

void drawMainScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(0, 0);
  display.print("HR: ");
  display.print(liveHR, 0);
  display.print(" bpm");

  display.setCursor(0, 8);
  display.print("Temp: ");
  display.print(liveBodyTemp, 1);
  display.print(" C");

  display.setCursor(0, 16);
  display.print("Pred: ");
  display.print(latestScore, 2);

  display.setCursor(0, 24);
  display.print(latestStress ? "STRESSED" : "NOT STRESSED");

  display.display();
}

void drawFeatureScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  // Abbreviated names to fit 128x32
  display.setCursor(0, 0);
  display.print("Bm:");
  display.print(feat_bvp_mean, 2);
  display.print(" Bs:");
  display.print(feat_bvp_std, 2);

  display.setCursor(0, 8);
  display.print("Tm:");
  display.print(feat_temp_mean, 2);
  display.print(" Ts:");
  display.print(feat_temp_std, 2);

  display.setCursor(0, 16);
  display.print("Tsl:");
  display.print(feat_temp_slope, 4);

  display.setCursor(0, 24);
  display.print("P:");
  display.print(latestScore, 2);
  display.print(latestStress ? " S" : " NS");

  display.display();
}

void setupModel() {
  model = tflite::GetModel(g_model);
  if (model == nullptr) {
    Serial.println("GetModel failed");
    while (true) delay(10);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize,
      error_reporter,
      nullptr,
      nullptr);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (true) delay(10);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model ready");
  Serial.print("Input type: "); Serial.println(input->type);
  Serial.print("Output type: "); Serial.println(output->type);
}

void setupDisplay() {
  Wire.begin(21, 22);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("SSD1306 init failed");
    while (true) delay(10);
  }

  display.clearDisplay();
  display.display();
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  randomSeed(micros());

  setupDisplay();
  setupModel();

  // Initial values
  liveHR = randFloat(70.0f, 90.0f);
  liveBodyTemp = randFloat(36.4f, 37.2f);

  runRandomInference();

  lastSensorUpdate = millis();
  lastInference = millis();
  lastScreenSwitch = millis();
}

void loop() {
  unsigned long now = millis();

  if (now - lastSensorUpdate >= SENSOR_UPDATE_MS) {
    lastSensorUpdate = now;
    updateLiveVitals();
  }

  if (now - lastInference >= INFERENCE_MS) {
    lastInference = now;
    runRandomInference();
  }

  if (now - lastScreenSwitch >= SCREEN_SWITCH_MS) {
    lastScreenSwitch = now;
    showMainScreen = !showMainScreen;
  }

  if (showMainScreen) {
    drawMainScreen();
  } else {
    drawFeatureScreen();
  }

  delay(40);
}