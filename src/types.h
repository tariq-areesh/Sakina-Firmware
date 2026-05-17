#pragma once
#include <Arduino.h>

// ---------------------------------------------------------------
// Command  — control messages exchanged over BLE
// ---------------------------------------------------------------
enum class Command { Start, Stop, Sync, Update };

// ---------------------------------------------------------------
// SimMode  — simulator control (sent as BLE strings)
//   "force:stress" -> ForceStress
//   "force:calm"   -> ForceCalm
//   "force:auto"   -> Auto
// ---------------------------------------------------------------
enum class SimMode { Auto, ForceStress, ForceCalm };

// ---------------------------------------------------------------
// Sample  — one timestamped sensor reading
// ---------------------------------------------------------------
struct Sample {
    uint32_t ts;    // millis() timestamp
    float    bvp;   // blood volume pulse (normalised [0.28, 0.82])
    float    hr;    // heart rate (bpm)
    float    temp;  // wrist skin temperature (°C)
};

// ---------------------------------------------------------------
// Window  — feature vector produced by SensorManager
//   tensor[5] = { BVP_mean, BVP_std, TEMP_mean, TEMP_std, TEMP_slope }
//   TEMP features are in normalised space: (T - 30) / 10
// ---------------------------------------------------------------
struct Window {
    uint32_t startTs;
    uint16_t durationSec;
    float    tensor[5];
};

// ---------------------------------------------------------------
// ModelPackage  — over-the-air weight bundle (future use)
// ---------------------------------------------------------------
struct ModelPackage {
    uint8_t* weights;
    uint32_t version;
    uint32_t checksum;
};
