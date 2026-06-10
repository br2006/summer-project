#include <SimpleFOC.h>
// #include <SD.h>


MagneticSensorI2C sensor = MagneticSensorI2C(AS5600_I2C);
BLDCMotor motor = BLDCMotor(11);
BLDCDriver3PWM driver = BLDCDriver3PWM(9, 10, 11);

// pendulum angle
float theta = 0;
float dtheta = 0;
String incomingBuffer = "";
bool receiving = false;

// angle read timing
unsigned long lastAngleRead = 0;
const unsigned long ANGLE_INTERVAL = 5000;    // 5ms in micros

// motor voltage
float target_voltage = 0;
float MAX_VOLTAGE = 12;

// derivative things
float lastAngle = 0.0;
float alpha = 0.1;
float t_old_derivative = micros();
float dt = 0.0;
float rawDtheta = 0.0;
float filteredDtheta = 0.0;

float integral = 0.0;
float t_old_integral = millis();
float dt_integral = 0.0;

unsigned long lastPrint = 0;


float initialAngle = 0;
bool calibrated = false;
const int CALIBRATION_SAMPLES = 20;   
int calibrationCount = 0;
float calibrationSum = 0;

Commander command = Commander(Serial);
void doTarget(char* cmd) {
    command.scalar(&target_voltage, cmd);
    target_voltage = constrain(target_voltage, -MAX_VOLTAGE, MAX_VOLTAGE);
}

float constrainAngle(float x){
    x = fmod(x + M_PI, _2PI);
    if (x < 0)
        x += _2PI;
    return x - M_PI;
}

float computeDerivative(float thet) {
  dt = micros() - t_old_derivative;
  t_old_derivative = micros();
  rawDtheta = (thet - lastAngle) / (dt / 1000);
  filteredDtheta = alpha * rawDtheta + (1.0 - alpha) * filteredDtheta;
  lastAngle = thet;
  return filteredDtheta;
}

void readPendulumAngle() {
  while (Serial1.available()) {
    char c = Serial1.read();

    if (c == '<') {
      incomingBuffer = "";
      receiving = true;

    } 
    else if (c == '>' && receiving) {
      float parsed = incomingBuffer.toFloat();
      if (!calibrated) {
        calibrationSum += parsed;
        calibrationCount++;
        Serial.print("Calibrating... sample ");
        Serial.print(calibrationCount);
        Serial.print("/");
        Serial.println(CALIBRATION_SAMPLES);

        if (calibrationCount >= CALIBRATION_SAMPLES) {
          initialAngle = calibrationSum / CALIBRATION_SAMPLES;
          calibrated = true;
          Serial.print("Calibration complete. Zero angle: ");
          Serial.println(initialAngle, 4);
        }

      } 
      else {
        // theta = fmod(parsed - initialAngle + M_PI, _2PI);
        // if (theta < 0) {theta += _2PI;}
        // theta -= M_PI;
        theta = parsed - initialAngle;
      }
      receiving = false;
    } 
    else if (receiving) {
      incomingBuffer += c;
    }
  }
}

void setup() {
    Serial.begin(115200);       // USB — Serial Monitor
    Serial1.begin(115200);      // link to UNO — RX1 on pin 19

    SimpleFOCDebug::enable(&Serial);

    sensor.init();
    motor.linkSensor(&sensor);

    driver.voltage_power_supply = 24;
    driver.init();
    motor.linkDriver(&driver);

    motor.voltage_sensor_align = 5;
    motor.foc_modulation = FOCModulationType::SpaceVectorPWM;
    motor.controller = MotionControlType::torque;
    motor.voltage_limit = MAX_VOLTAGE;

    motor.init();
    motor.initFOC();

    command.add('T', doTarget, "target voltage");

    Serial.println("Ready. Type T3 to set 3V, T0 to stop.");
    delay(1000);

}

float ki = 0;
float kd = -0;
float kp = -1000;

float control(float theta) {
  // theta = constrainAngle(theta);
  dt_integral = millis() - t_old_integral;
  // if (-0.7 < theta && theta < 0.7) {
  //   integral += theta * dt_integral / 1000;
  //   return kp * theta + kd * computeDerivative(theta) ;
  // }
  // else {integral = 0;}

  t_old_integral = millis();
  integral = constrain(integral, -12, 12);
  // Serial.println(integral);

  // return ki * integral + kd * computeDerivative(theta) + kp * theta;

  if (constrainAngle(theta) < 1.4 && constrainAngle(theta) > -1.4) {

    return kp*constrainAngle(theta);
  }
  return 4 * (theta + 3.1416);
  
  
  
}

float control1(float theta) {
  dt_integral = millis() - t_old_integral;
  
  // if (-0.7 < theta && theta < 0.7) {integral += theta * dt_integral / 1000;}
  // else {integral = 0;}
  t_old_integral = millis();
  integral = constrain(integral, -12, 12);
 
  return  -kp * theta;
  
}



void loop() {
    motor.loopFOC();
    // Serial.println(sensor.getAngle());


    if (micros() - lastAngleRead >= ANGLE_INTERVAL) {
        lastAngleRead = micros();
        readPendulumAngle();

    }

    // don't move motor until we have a reliable zero reference
    if (calibrated) {
      target_voltage = control(theta);
      // Serial.println(target_voltage);
      motor.move(target_voltage);
    } else {
        
        motor.move(0);
    }

    command.run();

    if (millis() - lastPrint >= 50) {
        lastPrint = millis();
        if (calibrated) {
            // Serial.print("Theta: ");
            // Serial.println(theta);
            // Serial.print(" rad  |  Target voltage: ");
            // Serial.println(target_voltage);
          // Serial.println(computeDerivative(sensor.getVelocity()));
            // Serial.println(theta);
          Serial.println(sensor.getVelocity());
          // Serial.println(computeDerivative(theta));
          // Serial.println(filter.deriv() - sensor.getVelocity());


            
        }
    }
}