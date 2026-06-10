#include <SimpleFOC.h>
#include<math.h>

// BLDCMotor(pole pair number);
BLDCMotor motor = BLDCMotor(11);
// BLDCDriver3PWM(pwmA, pwmB, pwmC, Enable(optional));
BLDCDriver3PWM driver = BLDCDriver3PWM(9, 10, 11); // set your pins

//sensor 
MagneticSensorI2C sensor = MagneticSensorI2C(AS5600_I2C);
int x = 0;
// float last = millis();
// instantiate the commander
Commander command = Commander(Serial);
void doTarget(char* cmd) { command.scalar(&motor.target, cmd); }
void doLimit(char* cmd) { command.scalar(&motor.voltage_limit, cmd); }

void setup() {

  // use monitoring with serial 
  Serial.begin(9600);
  // enable more verbose output for debugging
  // comment out if not needed
  SimpleFOCDebug::enable(&Serial);

  // driver config
  // power supply voltage [V]
  driver.voltage_power_supply = 20; //set your power supply voltage;
  // limit the maximal dc voltage the driver can set
  driver.voltage_limit = 10; //set your voltage limit; Usually half of power supply voltage is a good 
  if(!driver.init()){
    Serial.println("Driver init failed!");
    return;
  }
  // link the motor and the driver
  motor.linkDriver(&driver);

  // limiting motor movements
  // start very low for high resistance motors
  // current = voltage / resistance, so try to be well under 1Amp
  motor.voltage_limit = 10; //set your voltage limit in volts;   // [V]
 
  // open loop control config
  motor.controller = MotionControlType::velocity_openloop;

  // init motor hardware
  if(!motor.init()){
    Serial.println("Motor init failed!");
    return;
  }
  sensor.init();
  Serial.println("Sensor ready");
  // set the target velocity [rad/s]
  

  // add target command T
  command.add('T', doTarget, "target velocity");
  command.add('L', doLimit, "voltage limit");

  Serial.println("Motor ready!");
  Serial.println("Set target velocity [rad/s]");

  delay(1000);
}

// float speed() {
//   last = 5* (millis()) / 5000;
//   return sin(last) * 5;
// }


void loop() {
  
  motor.target = x; // one rotation per second

  // Serial.println(speed());

  // loop FOC algorithm, should be called as 
  // frequently as possible for best 
  // performance (e.g. 1kHz+)
  motor.loopFOC();

  // open loop velocity movement
  motor.move();
  unsigned long last = millis();
  // user communication
  command.run();
  sensor.update();
  if (millis() - last >= 500) {   // 20 Hz output
    last = millis();
    Serial.println(sensor.getVelocity(), 4);
    // Serial.println(x);
    x += 10;
    
  }
}
