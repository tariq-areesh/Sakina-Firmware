# Sakina | ESP32 Firmware

ESP32 firmware for the Sakina stress monitoring system. Runs a quantized TFLite MLP model on-device, simulates physiological sensor data, and streams inference results to the Flutter app over BLE.

Part of CPCS499 | Group C02 | King Abdulaziz University

---

## What It Does

The firmware handles the full edge pipeline:

1. Simulates BVP and wrist skin temperature at 1 Hz (replaces physical sensors until they are available)
2. Builds a 60-sample rolling window and extracts 5 features: BVP mean, BVP std, TEMP mean, TEMP std, TEMP slope
3. Standardizes the features using the same StandardScaler stats from training
4. Runs the quantized INT8 TFLite model on-device and classifies as Normal or Stressed
5. Broadcasts the result over BLE every 500 ms for the Flutter app to receive
6. Shows live readings on a 128x32 OLED display

---

## Files

```
src/
└── main.cpp          # Full firmware: simulator, inference, BLE, OLED

model/
├── model_data.h      # Model array declaration
└── model_data.cc     # Quantized INT8 TFLite model as a C array
```

---

## Hardware

| Component | Details |
|-----------|---------|
| Microcontroller | ESP32 |
| Display | SSD1306 OLED 128x32 (I2C) |
| BLE | Built-in ESP32 BLE stack |
| Sensors | Simulated (BVP + wrist temperature) |

OLED wired to SDA=21, SCL=22.

---

## Model

The deployed model is a 5-input MLP (5 -> 64 -> 32 -> 1) trained on a 10,000-subject synthetic cGAN dataset and the WESAD dataset. It was converted to TFLite and quantized to INT8 using representative data calibration.

Input features (in this exact order):

```
[BVP_mean, BVP_std, TEMP_mean, TEMP_std, TEMP_slope]
```

Temperature is normalized as `(raw_celsius - 30.0) / 10.0` before feature extraction. Features are then standardized with the scaler stats hardcoded in `main.cpp`.

To convert the model after retraining:

```bash
# In your training environment
python export_for_fl.py      # produces sakina_initial_weights.json
# Then requantize and run xxd to regenerate model_data.cc
xxd -i model.tflite > model_data.cc
```

---

## BLE Protocol

Device name: `Sakina ESP32`

| UUID | Role |
|------|------|
| `...90ab` | Service |
| `...90ac` | TX (notify) — firmware sends data here |
| `...90ad` | RX (write) — app sends commands here |

TX packet format (comma-separated):

```
stress:Normal,temp:36.8,hr:72,bvp:0.5123,score:0.312,advice:You are balanced...
```

RX commands (write to RX characteristic):

```
force:stress    # ramp simulator toward stressed state
force:calm      # ramp simulator toward calm state
force:auto      # return to random-walk mode
```

---

## Setup

Open the project in PlatformIO, build, and flash to your ESP32. Monitor at 115200 baud to see live feature values and inference results.

Make sure these libraries are installed:

```
Adafruit SSD1306
Adafruit GFX
TensorFlowLite_ESP32
ESP32 BLE Arduino
```