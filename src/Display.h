#pragma once
#include <Arduino.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ---------------------------------------------------------------
// Display  — SSD1306 128×32 OLED helper
//
//  Two alternating screens (toggled by main every 2.5 s):
//    drawMain()     — HR, Temp, Score, stress label
//    drawFeatures() — all 5 features + sim stress + score
// ---------------------------------------------------------------
class Display {
public:
    static constexpr int WIDTH  = 128;
    static constexpr int HEIGHT = 32;
    static constexpr uint8_t I2C_ADDR = 0x3C;

    bool begin();
    void drawMain(float hr, float tempC, float score, bool stressed);
    void drawFeatures(float bvpMean, float bvpStd,
                      float tempMean, float tempStd, float tempSlope,
                      float stressLevel, float score, bool stressed);

    static String stressText(bool stressed);
    static String adviceText(bool stressed, float score);

private:
    Adafruit_SSD1306 _oled{WIDTH, HEIGHT, &Wire, -1};
};
