#pragma once
#include "types.h"
#include <Arduino.h>

// ---------------------------------------------------------------
// SensorManager  — sensor simulation, ring buffers, feature extraction
//
//  Typical call sequence each 1-Hz tick from main loop:
//    1. updateStress(now)      — drift latent stress level every 5 s
//    2. pushBvpSample(now)     — generate BVP, update HR, push to ring buffer
//    3. pushTempSample(now)    — generate TEMP, low-pass filter, normalise, push
//    4. if buffersReady():
//         Window w = nextWindow()   — compute 5 features + log to Serial
//         float score = model.predict(w)
//
//  Simulator control:
//    handleCommand("force:stress" | "force:calm" | "force:auto")
// ---------------------------------------------------------------
class SensorManager {
public:
    static constexpr int   BUFFER_SIZE  = 60;
    static constexpr float WINDOW_SECS  = 60.0f;
    static constexpr float TEMP_ALPHA   = 0.10f;  // low-pass smoothing for TEMP
    static constexpr float STRESS_DRIFT_SEC = 5.0f;

    bool   begin();
    void   updateStress(unsigned long nowMs);
    void   pushBvpSample(unsigned long nowMs);
    void   pushTempSample(unsigned long nowMs);
    Window nextWindow();
    bool   buffersReady() const;
    void   handleCommand(const String& cmd);

    // Getters used by BLE and Display
    float getHr()          const { return _currentHr; }
    float getRawTempC()    const { return _rawTempC;  }
    float getSmoothedBvp() const { return _smoothedBvp; }
    float getStressLevel() const { return _stressLevel; }

    // Feature cache — populated by the last nextWindow() call
    float featBvpMean   = 0.0f;
    float featBvpStd    = 0.0f;
    float featTempMean  = 0.0f;
    float featTempStd   = 0.0f;
    float featTempSlope = 0.0f;

private:
    // Ring buffers
    float _bvpBuf[BUFFER_SIZE]  = {};
    float _tempBuf[BUFFER_SIZE] = {};
    int   _bvpIdx  = 0;
    int   _tempIdx = 0;
    bool  _bvpFull  = false;
    bool  _tempFull = false;

    // Simulator state
    SimMode _mode          = SimMode::Auto;
    float   _stressLevel   = 0.0f;   // 0 = calm .. 1 = fully stressed
    float   _simHr         = 72.0f;
    float   _currentHr     = 72.0f;
    float   _rawTempC      = 36.8f;
    float   _smoothedTemp  = 36.8f;
    float   _smoothedBvp   = 0.50f;
    bool    _tempFilterInit = false;

    unsigned long _lastDriftMs = 0;

    // Signal generation
    float _generateBvp(unsigned long nowMs);
    float _generateTemp(unsigned long nowMs);

    // Buffer helpers
    void  _pushBvp(float v);
    void  _pushTemp(float v);
    void  _copyOrdered(const float* src, int size, int idx, bool full, float* dst) const;

    // Math helpers
    static float _mean(const float* buf, int n);
    static float _std(const float* buf, int n, float m);
    static float _lowPass(float in, float prev, float alpha, bool& init);
    static float _randFloat(float lo, float hi);
    static float _clamp(float v, float lo, float hi);
    static float _normTempC(float t);
};
