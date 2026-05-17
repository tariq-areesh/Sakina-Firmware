#pragma once
#include "types.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------------------------------------------------------------
// Model  — wraps the quantised TFLite stress-classification model
//
//  Pipeline (must match Data_prep.py + training_arc.py exactly):
//    1. Receive a Window whose tensor[5] holds pre-computed features
//       { BVP_mean, BVP_std, TEMP_mean, TEMP_std, TEMP_slope }
//    2. Apply StandardScaler  z = (x - SCALER_MEAN) / SCALER_SCALE
//    3. INT8-quantise using TFLite tensor params
//    4. Invoke interpreter  →  dequantise output sigmoid [0, 1]
//    5. label = (score >= STRESS_THRESHOLD)
// ---------------------------------------------------------------
class Model {
public:
    static constexpr int   N_FEATURES       = 5;
    static constexpr float STRESS_THRESHOLD = 0.5f;

    // StandardScaler stats from training NPZ
    // Order: BVP_mean, BVP_std, TEMP_mean, TEMP_std, TEMP_slope
    static const float SCALER_MEAN[N_FEATURES];
    static const float SCALER_SCALE[N_FEATURES];

    bool  load(const uint8_t* storagePtr);
    float predict(const Window& w);
    bool  applyGlobal(const ModelPackage& pkg);

private:
    static tflite::MicroErrorReporter _microReporter;
    static tflite::ErrorReporter*     _reporter;

    const tflite::Model*      _tflModel    = nullptr;
    tflite::MicroInterpreter* _interpreter = nullptr;
    TfLiteTensor*             _input       = nullptr;
    TfLiteTensor*             _output      = nullptr;

    static constexpr size_t kArenaSize = 32 * 1024;
    uint8_t _arena[kArenaSize];

    int8_t _quantize(float x, float scale, int zp);
    float  _dequantize(int8_t x, float scale, int zp);
};
