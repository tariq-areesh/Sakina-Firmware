#include "BLELink.h"

// ---------------------------------------------------------------
// UUID constants
// ---------------------------------------------------------------
const char* BLELink::SERVICE_UUID = "12345678-1234-1234-1234-1234567890ab";
const char* BLELink::TX_UUID      = "12345678-1234-1234-1234-1234567890ac";
const char* BLELink::RX_UUID      = "12345678-1234-1234-1234-1234567890ad";

// ---------------------------------------------------------------
// BLE callbacks  — defined here so they can access BLELink fields
// ---------------------------------------------------------------
class _ServerCB : public BLEServerCallbacks {
public:
    explicit _ServerCB(BLELink* link) : _link(link) {}
    void onConnect(BLEServer*) override {
        _link->isConnected = true;
        Serial.println("[BLE] connected");
    }
    void onDisconnect(BLEServer*) override {
        _link->isConnected = false;
        Serial.println("[BLE] disconnected");
    }
private:
    BLELink* _link;
};

class _RxCB : public BLECharacteristicCallbacks {
public:
    explicit _RxCB(BLELink* link) : _link(link) {}
    void onWrite(BLECharacteristic* c) override {
        std::string v = c->getValue();
        if (!v.empty() && !_link->_cmdReady) {
            strncpy(_link->_cmdBuf, v.c_str(), sizeof(_link->_cmdBuf) - 1);
            _link->_cmdBuf[sizeof(_link->_cmdBuf) - 1] = '\0';
            _link->_cmdReady = true;
        }
    }
private:
    BLELink* _link;
};

// ---------------------------------------------------------------
// begin()  — initialise BLE device, GATT service, and advertise
// ---------------------------------------------------------------
void BLELink::begin() {
    BLEDevice::init(deviceName.c_str());
    _pServer = BLEDevice::createServer();
    _pServer->setCallbacks(new _ServerCB(this));

    BLEService* svc = _pServer->createService(SERVICE_UUID);

    _pTx = svc->createCharacteristic(TX_UUID,
        BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ);
    _pTx->addDescriptor(new BLE2902());

    _pRx = svc->createCharacteristic(RX_UUID,
        BLECharacteristic::PROPERTY_WRITE |
        BLECharacteristic::PROPERTY_WRITE_NR |
        BLECharacteristic::PROPERTY_READ);
    _pRx->setCallbacks(new _RxCB(this));

    svc->start();
    advertise();
}

void BLELink::advertise() {
    BLEAdvertising* adv = BLEDevice::getAdvertising();
    adv->addServiceUUID(SERVICE_UUID);
    adv->setScanResponse(true);
    adv->start();
    Serial.println("[BLE] advertising started");
}

// ---------------------------------------------------------------
// sendStatus()  — build and notify the TX packet
// ---------------------------------------------------------------
bool BLELink::sendStatus(uint8_t label, float conf, uint32_t ts,
                          float tempC, float hr, float bvp, const String& advice) {
    if (!isConnected || !_pTx) return false;

    String p;
    p += "stress:" + String(label ? "Stressed" : "Normal");
    p += ",temp:"  + String(tempC, 1);
    p += ",hr:"    + String((int)roundf(hr));
    p += ",bvp:"   + String(bvp, 4);
    p += ",score:" + String(conf, 3);
    p += ",advice:"+ advice;

    _pTx->setValue(p.c_str());
    _pTx->notify();
    Serial.print("[BLE] TX -> "); Serial.println(p);
    return true;
}

// ---------------------------------------------------------------
// hasCommand() / recvCommand()  — consume BLE RX outside ISR
// ---------------------------------------------------------------
bool BLELink::hasCommand() const {
    return _cmdReady;
}

String BLELink::recvCommand() {
    String cmd(_cmdBuf);
    _cmdReady = false;
    return cmd;
}

// ---------------------------------------------------------------
// poll()  — restart advertising after a disconnect
// ---------------------------------------------------------------
void BLELink::poll() {
    if (!isConnected && _oldConn) {
        delay(300);
        _pServer->startAdvertising();
        Serial.println("[BLE] re-advertising");
        _oldConn = false;
    }
    if (isConnected && !_oldConn) _oldConn = true;
}
