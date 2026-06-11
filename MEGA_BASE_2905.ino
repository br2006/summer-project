#include <SimpleFOC.h>

MagneticSensorI2C sensor = MagneticSensorI2C(AS5600_I2C);
BLDCMotor motor = BLDCMotor(11);
BLDCDriver3PWM driver = BLDCDriver3PWM(9, 10, 11);

// pendulum angle
float theta = 0;
String incomingBuffer = "";
bool receiving = false;

// angle read timing
unsigned long lastAngleRead = 0;
const unsigned long ANGLE_INTERVAL = 5000;    // 5ms in micros

// motor voltage
float target_voltage = 0;
float MAX_VOLTAGE = 10.0;

// serial print timing
unsigned long lastPrint = 0;

//initial angle calculation variables
// add these variables at the top with your other globals
float initialAngle = 0;
bool calibrated = false;
const int CALIBRATION_SAMPLES = 20;    // number of readings to average
int calibrationCount = 0;
float calibrationSum = 0;

Commander command = Commander(Serial);
void doTarget(char* cmd) {
    command.scalar(&target_voltage, cmd);
    target_voltage = constrain(target_voltage, -MAX_VOLTAGE, MAX_VOLTAGE);
}

void readPendulumAngle() {
    while (Serial1.available()) {
        char c = Serial1.read();

        if (c == '<') {
            incomingBuffer = "";
            receiving = true;

        } else if (c == '>' && receiving) {
            float parsed = incomingBuffer.toFloat();

            if (parsed >= -6.0 && parsed <= 6.0) {

                if (!calibrated) {
                    // still collecting calibration samples
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

                } else {
                    // normal operation — subtract initial angle
                    theta = parsed - initialAngle;
                }
            } else {
                Serial.print("Rejected: ");
                Serial.println(parsed);
            }
            receiving = false;

        } else if (receiving) {
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

    driver.voltage_power_supply = 14;
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


void loop() {
    motor.loopFOC();

    if (micros() - lastAngleRead >= ANGLE_INTERVAL) {
        lastAngleRead = micros();
        readPendulumAngle();
    }

    // don't move motor until we have a reliable zero reference
    if (calibrated) {
        motor.move(target_voltage);
    } else {
        motor.move(0);
    }

    command.run();

    if (millis() - lastPrint >= 500) {
        lastPrint = millis();
        if (calibrated) {
            Serial.print("Theta: ");
            Serial.print(theta, 4);
            Serial.print(" rad  |  Target voltage: ");
            Serial.println(target_voltage, 2);
        }
    }
}