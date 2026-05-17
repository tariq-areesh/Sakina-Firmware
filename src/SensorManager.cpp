#include "SensorManager.h"
#include <math.h>

// ---------------------------------------------------------------
// begin()  — initialise buffers with neutral values so the first
//            inference fires after one full window rather than 60 s
// ---------------------------------------------------------------
bool SensorManager::begin() {
    // Pre-fill at the decision boundary (normalised 0.57 ≈ 35.7 °C)
    // so stressed or calm samples push the window mean quickly.
    constexpr float neutralNorm = 0.57f;
    for (int i = 0; i < BUFFER_SIZE; i++) _bvpBuf[i]  = 0.51f;
    for (int i = 0; i < BUFFER_SIZE; i++) _tempBuf[i] = neutralNorm;
    _bvpIdx  = _tempIdx  = 0;
    _bvpFull = _tempFull = false;

    _rawTempC     = _randFloat(36.5f, 37.0f);
    _smoothedTemp = _rawTempC;
    _simHr        = _randFloat(62.0f, 75.0f);
    _currentHr    = _simHr;
    _stressLevel  = 0.0f;
    return true;
}

// ---------------------------------------------------------------
// updateStress()  — drift the latent stress level every 5 s
// ---------------------------------------------------------------
void SensorManager::updateStress(unsigned long nowMs) {
    if (nowMs - _lastDriftMs < (unsigned long)(STRESS_DRIFT_SEC * 1000)) return;
    _lastDriftMs = nowMs;

    if      (_mode == SimMode::ForceStress) _stressLevel = _clamp(_stressLevel + 0.10f, 0.0f, 1.0f);
    else if (_mode == SimMode::ForceCalm)   _stressLevel = _clamp(_stressLevel - 0.10f, 0.0f, 1.0f);
    else {
        _stressLevel += _randFloat(-0.08f, 0.08f);
        _stressLevel  = _clamp(_stressLevel, 0.0f, 1.0f);
    }
}

// ---------------------------------------------------------------
// pushBvpSample()  — update HR toward stress target, generate BVP
// ---------------------------------------------------------------
void SensorManager::pushBvpSample(unsigned long nowMs) {
    // HR rises with stress: calm ~68 bpm, stressed ~106 bpm
    const float hrTarget = 68.0f + _stressLevel * 38.0f;
    _simHr += (hrTarget - _simHr) * 0.05f + _randFloat(-0.3f, 0.3f);
    _simHr  = _clamp(_simHr, 55.0f, 115.0f);
    _currentHr = _simHr;

    // No LPF on BVP at 1 Hz — consecutive samples land at different cardiac
    // phases, giving natural variance that matches training BVP_std (~0.063).
    const float raw = _generateBvp(nowMs);
    _smoothedBvp = raw;
    _pushBvp(raw);
}

// ---------------------------------------------------------------
// pushTempSample()  — generate TEMP, low-pass filter, normalise
// ---------------------------------------------------------------
void SensorManager::pushTempSample(unsigned long nowMs) {
    const float raw = _generateTemp(nowMs);
    _smoothedTemp = _lowPass(raw, _smoothedTemp, TEMP_ALPHA, _tempFilterInit);
    _rawTempC     = _smoothedTemp;
    _pushTemp(_normTempC(_rawTempC));  // push normalised value matching training
}

// ---------------------------------------------------------------
// nextWindow()  — extract ordered buffers, compute 5 features, log
// ---------------------------------------------------------------
Window SensorManager::nextWindow() {
    static float ordBvp[BUFFER_SIZE];
    static float ordTemp[BUFFER_SIZE];
    _copyOrdered(_bvpBuf,  BUFFER_SIZE, _bvpIdx,  _bvpFull,  ordBvp);
    _copyOrdered(_tempBuf, BUFFER_SIZE, _tempIdx, _tempFull, ordTemp);

    featBvpMean   = _mean(ordBvp,  BUFFER_SIZE);
    featBvpStd    = _std (ordBvp,  BUFFER_SIZE, featBvpMean);
    featTempMean  = _mean(ordTemp, BUFFER_SIZE);
    featTempStd   = _std (ordTemp, BUFFER_SIZE, featTempMean);
    // Matches Data_prep.py: (last - first) / WINDOW_SIZE
    featTempSlope = (ordTemp[BUFFER_SIZE - 1] - ordTemp[0]) / WINDOW_SECS;

    Serial.println("========================================");
    Serial.print("SimStressLevel:  "); Serial.println(_stressLevel, 3);
    Serial.print("Wrist Temp raw:  "); Serial.print(_rawTempC, 2); Serial.println(" C");
    Serial.print("Wrist Temp norm: "); Serial.println(_normTempC(_rawTempC), 4);
    Serial.print("HR:              "); Serial.print(_currentHr, 1); Serial.println(" bpm");
    Serial.print("BVP_mean:        "); Serial.println(featBvpMean, 5);
    Serial.print("BVP_std:         "); Serial.println(featBvpStd, 5);
    Serial.print("TEMP_mean(norm): "); Serial.println(featTempMean, 5);
    Serial.print("TEMP_std (norm): "); Serial.println(featTempStd, 5);
    Serial.print("TEMP_slope:      "); Serial.println(featTempSlope, 6);

    Window w;
    w.startTs      = millis();
    w.durationSec  = (uint16_t)WINDOW_SECS;
    w.tensor[0]    = featBvpMean;
    w.tensor[1]    = featBvpStd;
    w.tensor[2]    = featTempMean;
    w.tensor[3]    = featTempStd;
    w.tensor[4]    = featTempSlope;
    return w;
}

bool SensorManager::buffersReady() const {
    return _bvpFull && _tempFull;
}

// ---------------------------------------------------------------
// handleCommand()  — parse BLE simulator control string
// ---------------------------------------------------------------
void SensorManager::handleCommand(const String& cmd) {
    Serial.print("[SensorManager] cmd: "); Serial.println(cmd);
    if      (cmd == "force:stress") { _mode = SimMode::ForceStress; Serial.println("  -> forcing stress"); }
    else if (cmd == "force:calm")   { _mode = SimMode::ForceCalm;   Serial.println("  -> forcing calm");   }
    else if (cmd == "force:auto")   { _mode = SimMode::Auto;        Serial.println("  -> auto mode");      }
    else                              Serial.println("  -> unknown (use force:stress/calm/auto)");
}

// ---------------------------------------------------------------
// Signal generators
// ---------------------------------------------------------------
float SensorManager::_generateBvp(unsigned long nowMs) {
    const float beatMs = 60000.0f / max(_currentHr, 1.0f);
    const float phase  = fmod((float)nowMs, beatMs) / beatMs;
    float pulse;
    if      (phase < 0.12f) pulse = phase / 0.12f;
    else if (phase < 0.28f) pulse = 1.0f - ((phase - 0.12f) / 0.16f) * 0.52f;
    else                    pulse = 0.48f - ((phase - 0.28f) / 0.72f) * 0.10f;

    const float noiseAmp = 0.010f + _stressLevel * 0.015f;
    return _clamp(
        0.42f + pulse * 0.22f
              + sinf((float)nowMs * 0.0012f)   * 0.010f
              + sinf((float)nowMs * 0.00045f)  * 0.018f
              + _randFloat(-noiseAmp, noiseAmp),
        0.28f, 0.82f);
}

// Wrist skin temp drops with stress (peripheral vasoconstriction).
// calm 36.8 C (norm 0.68), stressed 31.5 C (norm 0.15)
// Decision boundary ~35.7 C (norm 0.57)
float SensorManager::_generateTemp(unsigned long nowMs) {
    const float target = 36.8f + (31.5f - 36.8f) * _stressLevel;
    return _clamp(
        target + sinf((float)nowMs * 0.000015f) * 0.10f
               + _randFloat(-0.05f, 0.05f),
        30.0f, 39.5f);
}

// ---------------------------------------------------------------
// Ring buffer helpers
// ---------------------------------------------------------------
void SensorManager::_pushBvp(float v) {
    _bvpBuf[_bvpIdx] = v;
    _bvpIdx = (_bvpIdx + 1) % BUFFER_SIZE;
    if (_bvpIdx == 0) _bvpFull = true;
}

void SensorManager::_pushTemp(float v) {
    _tempBuf[_tempIdx] = v;
    _tempIdx = (_tempIdx + 1) % BUFFER_SIZE;
    if (_tempIdx == 0) _tempFull = true;
}

void SensorManager::_copyOrdered(const float* src, int size, int idx,
                                  bool full, float* dst) const {
    if (!full) {
        for (int i = 0; i < idx; i++) dst[i] = src[i];
        return;
    }
    int k = 0;
    for (int i = idx;  i < size; i++) dst[k++] = src[i];
    for (int i = 0;    i < idx;  i++) dst[k++] = src[i];
}

// ---------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------
float SensorManager::_mean(const float* buf, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += buf[i];
    return s / n;
}

// Population std (ddof=0) — matches np.nanstd in Data_prep.py
float SensorManager::_std(const float* buf, int n, float m) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) { float d = buf[i] - m; acc += d * d; }
    return sqrtf(acc / n);
}

float SensorManager::_lowPass(float in, float prev, float alpha, bool& init) {
    if (!init) { init = true; return in; }
    return alpha * in + (1.0f - alpha) * prev;
}

float SensorManager::_randFloat(float lo, float hi) {
    return lo + ((float)random(0, 1000000) / 999999.0f) * (hi - lo);
}

float SensorManager::_clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Normalise raw wrist skin temp to match training feature space.
// Derived by reverse-mapping NPZ TEMP_mean class distributions:
//   normal 0.683 → 36.83 C, stress 0.458 → 34.58 C  →  (T - 30) / 10
float SensorManager::_normTempC(float t) {
    return (t - 30.0f) / 10.0f;
}
