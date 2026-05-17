#include "Model.h"
#include <Arduino.h>
#include <math.h>

// ---------------------------------------------------------------
// Static member definitions
// ---------------------------------------------------------------
tflite::MicroErrorReporter Model::_microReporter;
tflite::ErrorReporter*     Model::_reporter = &Model::_microReporter;

// Derived from 10000_subj_synthetic_cGAN_training_ready_noHR.npz
// Replace with best_thr from training_arc.py output if you retrain.
const float Model::SCALER_MEAN[Model::N_FEATURES]  = { 0.5117f,  0.0625f,  0.6177f,  0.0890f, -0.0000203f };
const float Model::SCALER_SCALE[Model::N_FEATURES] = { 0.0317f,  0.0338f,  0.2587f,  0.1071f,  0.00471f   };

// ---------------------------------------------------------------
// load()  — parse and allocate TFLite model from flash byte array
// ---------------------------------------------------------------
bool Model::load(const uint8_t* storagePtr) {
    _tflModel = tflite::GetModel(storagePtr);
    if (!_tflModel) {
        Serial.println("[Model] GetModel failed");
        return false;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter staticInterp(
        _tflModel, resolver, _arena, kArenaSize, _reporter, nullptr, nullptr);
    _interpreter = &staticInterp;

    if (_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[Model] AllocateTensors failed");
        return false;
    }
    _input  = _interpreter->input(0);
    _output = _interpreter->output(0);

    if (_input->dims->data[_input->dims->size - 1] != N_FEATURES)
        Serial.println("[Model] WARN: input size mismatch — check N_FEATURES");

    Serial.println("[Model] ready");
    Serial.print("  in  scale/zp: "); Serial.print(_input->params.scale, 8);
    Serial.print(" / ");              Serial.println(_input->params.zero_point);
    Serial.print("  out scale/zp: "); Serial.print(_output->params.scale, 8);
    Serial.print(" / ");              Serial.println(_output->params.zero_point);
    return true;
}

// ---------------------------------------------------------------
// predict()  — run one inference on a pre-computed feature Window
// ---------------------------------------------------------------
float Model::predict(const Window& w) {
    const float inScale = _input->params.scale;
    const int   inZp    = _input->params.zero_point;

    for (int i = 0; i < N_FEATURES; i++) {
        const float z = (w.tensor[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
        _input->data.int8[i] = _quantize(z, inScale, inZp);
    }

    if (_interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[Model] ERROR: Invoke failed");
        return NAN;
    }
    return _dequantize(_output->data.int8[0],
                       _output->params.scale,
                       _output->params.zero_point);
}

// ---------------------------------------------------------------
// applyGlobal()  — reload model from an OTA package (future use)
// ---------------------------------------------------------------
bool Model::applyGlobal(const ModelPackage& pkg) {
    return load(pkg.weights);
}

// ---------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------
int8_t Model::_quantize(float x, float scale, int zp) {
    int32_t q = (int32_t)roundf(x / scale) + zp;
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    return (int8_t)q;
}

float Model::_dequantize(int8_t x, float scale, int zp) {
    return ((int)x - zp) * scale;
}
