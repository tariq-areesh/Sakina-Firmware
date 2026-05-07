#include <Arduino.h>
#include <math.h>
#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

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

int8_t quantize_int8(float x, float scale, int zero_point) {
  int32_t q = (int32_t)roundf(x / scale) + zero_point;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (int8_t)q;
}

float dequantize_int8(int8_t x, float scale, int zero_point) {
  return ((int)x - zero_point) * scale;
}

float randFloat(float minVal, float maxVal) {
  long r = random(0, 1000000);
  float t = (float)r / 999999.0f;
  return minVal + t * (maxVal - minVal);
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
  float y = dequantize_int8(raw, output->params.scale, output->params.zero_point);
  return y;
}

void print_inference(float bvp_mean, float bvp_std, float temp_mean, float temp_std, float temp_slope, float score) {
  Serial.println("========================================");
  Serial.println("Random inference sample");
  Serial.print("BVP_mean   : "); Serial.println(bvp_mean, 6);
  Serial.print("BVP_std    : "); Serial.println(bvp_std, 6);
  Serial.print("TEMP_mean  : "); Serial.println(temp_mean, 6);
  Serial.print("TEMP_std   : "); Serial.println(temp_std, 6);
  Serial.print("TEMP_slope : "); Serial.println(temp_slope, 6);
  Serial.print("Prediction : "); Serial.println(score, 6);

  if (score >= 0.5f) {
    Serial.println("Class      : STRESSED");
  } else {
    Serial.println("Class      : NOT STRESSED");
  }
}

void run_random_inference() {
  float bvp_mean   = randFloat(BVP_MEAN_MIN,   BVP_MEAN_MAX);
  float bvp_std    = randFloat(BVP_STD_MIN,    BVP_STD_MAX);
  float temp_mean  = randFloat(TEMP_MEAN_MIN,  TEMP_MEAN_MAX);
  float temp_std   = randFloat(TEMP_STD_MIN,   TEMP_STD_MAX);
  float temp_slope = randFloat(TEMP_SLOPE_MIN, TEMP_SLOPE_MAX);

  float score = run_model(bvp_mean, bvp_std, temp_mean, temp_std, temp_slope);
  print_inference(bvp_mean, bvp_std, temp_mean, temp_std, temp_slope, score);
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("boot");

  randomSeed(micros());

  model = tflite::GetModel(g_model);
  if (model == nullptr) {
    Serial.println("GetModel failed");
    return;
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
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model ready");
  Serial.print("Input scale: ");
  Serial.println(input->params.scale, 8);
  Serial.print("Input zero point: ");
  Serial.println(input->params.zero_point);
  Serial.print("Output scale: ");
  Serial.println(output->params.scale, 8);
  Serial.print("Output zero point: ");
  Serial.println(output->params.zero_point);
}

void loop() {
  run_random_inference();
  delay(3000);
}