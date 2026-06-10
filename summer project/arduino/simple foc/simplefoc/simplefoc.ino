#include <Wire.h>
#include <SimpleFOC.h>

MagneticSensorI2C sensor = MagneticSensorI2C(AS5600_I2C);

void setup() {
  Serial.begin(9600);
  Wire.begin();
  Wire.setClock(100000);

  delay(200);     // let serial settle a bit
  sensor.init();
  Serial.println("Sensor ready");
}

void loop() {
  sensor.update();

  static unsigned long last = 0;
  if (millis() - last >= 100) {   // 10 Hz output
    last = millis();
    Serial.println(sensor.getVelocity(), 4);
    // Serial.println(sensor.getVelocity(), 4);
  }
}