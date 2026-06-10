#include <Wire.h>
#include <SimpleFOC.h>

MagneticSensorI2C sensor = MagneticSensorI2C(AS5600_I2C);

unsigned long lastSend = 0;
const int SEND_INTERVAL = 5;    // 5ms = 200Hz
byte buffer[4];

void setup() {
    Serial.begin(115200);       // hardware serial TX on pin 1
    Wire.begin();
    Wire.setClock(400000);      // 400kHz I2C
    delay(200);
    sensor.init();


    
    
}

void loop() {
    sensor.update();

    if (millis() - lastSend >= SEND_INTERVAL) {
      float a = sensor.getAngle();
      Serial.println(a);
      lastSend = millis();


      memcpy(buffer, &a, sizeof(a));
      Serial.write(buffer, sizeof(buffer));
        
    }
}