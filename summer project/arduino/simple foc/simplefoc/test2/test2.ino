#include <SimpleFOC.h>

// BLDCDriver3PWM(pwmA, pwmB, pwmC, (en optional))
BLDCDriver3PWM driver = BLDCDriver3PWM(9,10,11); // set your pins

void setup() {
  
  // use monitoring with serial 
  Serial.begin(9600);
  // enable more verbose output for debugging
  // comment out if not needed
  SimpleFOCDebug::enable(&Serial);
  
  // power supply voltage [V]
  driver.voltage_power_supply = 22;
  // Max DC voltage allowed - default voltage_power_supply
  driver.voltage_limit = 24;

  // driver init
  if (!driver.init()){
    Serial.println("Driver init failed!");
    return;
  }

  // enable driver
  driver.enable();
  Serial.println("Driver ready!");
  delay(1000);
}

void loop() {
    // setting pwm
    // phase A: 3V
    // phase B: 6V
    // phase C: 5V
    driver.setPwm(3,6,5);
}
