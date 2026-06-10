#include <SoftwareSerial.h>
#include <Wire.h>
#include <SimpleFOC.h>

MagneticSensorI2C sensor = MagneticSensorI2C(AS5600_I2C);

SoftwareSerial link(10, 11); // RX unused, TX=11

void setup() {
  Serial.begin(115200);
  link.begin(115200);
  Serial.println("UNO sending...");


  Wire.begin();
  Wire.setClock(100000);

  delay(200);     // let serial settle a bit
  sensor.init();
  Serial.println("Sensor ready");

}

void loop() {
  while (true) {
    sensor.update();
  // link.println(sensor.getAngle());
  
    // link.write('\n');
    float value = sensor.getAngle();
    

    if (2 < value < 3) {link.println(value);}
    // link.write((byte*)&value, sizeof(value)); // Send as binary
  }
  

  // link.println(millis());
  // Serial.println(sensor.getAngle());
  // delay(500);
}
