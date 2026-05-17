#pragma once
#include "types.h"

#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>

// ---------------------------------------------------------------
// BLELink  — manages the BLE GATT server for Sakina ESP32
//
//  UUIDs:
//    Service  12345678-1234-1234-1234-1234567890ab
//    TX char  ...90ac   NOTIFY | READ   (device → phone)
//    RX char  ...90ad   WRITE           (phone  → device)
//
//  TX packet format (comma-separated):
//    stress:<Normal|Stressed>,temp:<°C>,hr:<bpm>,bvp:<val>,
//    score:<sigmoid>,advice:<text>
//
//  RX commands:
//    "force:stress" | "force:calm" | "force:auto"
// ---------------------------------------------------------------
class BLELink {
public:
    String   deviceName = "Sakina ESP32";
    bool     isConnected = false;
    uint16_t mtu         = 512;

    void begin();
    void advertise();
    bool sendStatus(uint8_t label, float conf, uint32_t ts,
                    float tempC, float hr, float bvp, const String& advice);
    bool hasCommand() const;
    String recvCommand();
    void poll();  // call every loop() to handle reconnect

private:
    static const char* SERVICE_UUID;
    static const char* TX_UUID;
    static const char* RX_UUID;

    BLEServer*         _pServer  = nullptr;
    BLECharacteristic* _pTx      = nullptr;
    BLECharacteristic* _pRx      = nullptr;
    bool               _oldConn  = false;
    volatile bool      _cmdReady = false;
    char               _cmdBuf[64] = {0};

    friend class _ServerCB;
    friend class _RxCB;
};
