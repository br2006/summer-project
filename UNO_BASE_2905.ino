#include <Wire.h>
#include <SimpleFOC.h>

MagneticSensorI2C sensor = MagneticSensorI2C(AS5600_I2C);

unsigned long lastSend = 0;
const unsigned long SEND_INTERVAL = 5;    // ms

void setup() {
    Serial.begin(115200);
    Wire.begin();
    Wire.setClock(400000);
    delay(200);
    sensor.init();
}

void loop() {
    sensor.update();

    if (millis() - lastSend >= SEND_INTERVAL) {
        lastSend = millis();
        Serial.print('<');
        Serial.print(sensor.getAngle(), 4);
        Serial.print('>');
    }
}