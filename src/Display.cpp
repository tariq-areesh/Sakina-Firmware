#include "Display.h"
#include <Wire.h>

// ---------------------------------------------------------------
// begin()  — I2C on GPIO 21/22, init SSD1306
// ---------------------------------------------------------------
bool Display::begin() {
    Wire.begin(21, 22);
    if (!_oled.begin(SSD1306_SWITCHCAPVCC, I2C_ADDR)) {
        Serial.println("[Display] SSD1306 init failed");
        return false;
    }
    _oled.clearDisplay();
    _oled.display();
    return true;
}

// ---------------------------------------------------------------
// drawMain()  — HR / Temp / Score / stress label
// ---------------------------------------------------------------
void Display::drawMain(float hr, float tempC, float score, bool stressed) {
    _oled.clearDisplay();
    _oled.setTextSize(1);
    _oled.setTextColor(SSD1306_WHITE);
    _oled.setCursor(0,  0); _oled.print("HR: ");    _oled.print(hr, 0);    _oled.print(" bpm");
    _oled.setCursor(0,  8); _oled.print("Temp: ");  _oled.print(tempC, 1); _oled.print(" C");
    _oled.setCursor(0, 16); _oled.print("Score: "); _oled.print(score, 3);
    _oled.setCursor(0, 24); _oled.print(stressed ? "STRESSED" : "NOT STRESSED");
    _oled.display();
}

// ---------------------------------------------------------------
// drawFeatures()  — all 5 extracted features + sim level
// ---------------------------------------------------------------
void Display::drawFeatures(float bvpMean, float bvpStd,
                            float tempMean, float tempStd, float tempSlope,
                            float stressLevel, float score, bool stressed) {
    _oled.clearDisplay();
    _oled.setTextSize(1);
    _oled.setTextColor(SSD1306_WHITE);
    _oled.setCursor(0,  0); _oled.print("Bm:"); _oled.print(bvpMean, 2);
                             _oled.print(" Bs:"); _oled.print(bvpStd, 2);
    _oled.setCursor(0,  8); _oled.print("Tm:"); _oled.print(tempMean, 3);
                             _oled.print(" Ts:"); _oled.print(tempStd, 3);
    _oled.setCursor(0, 16); _oled.print("Sl:"); _oled.print(tempSlope, 4);
                             _oled.print(" SL:"); _oled.print(stressLevel, 2);
    _oled.setCursor(0, 24); _oled.print("P:"); _oled.print(score, 3);
                             _oled.print(stressed ? " STR" : " NRM");
    _oled.display();
}

// ---------------------------------------------------------------
// Static helpers — used by BLELink and Display
// ---------------------------------------------------------------
String Display::stressText(bool stressed) {
    return stressed ? "Stressed" : "Normal";
}

String Display::adviceText(bool stressed, float score) {
    if (!stressed)          return "You are balanced. Maintain your current routine.";
    if (score < 0.65f)      return "Slight stress. Try slow breathing for one minute.";
    if (score < 0.80f)      return "Moderate stress. Take a short walk or rest break.";
    return "High stress. Sit down and focus on slow exhales.";
}
