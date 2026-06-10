#include <PID_v1.h>

#define PIN_INPUT  A2
#define PIN_OUTPUT 3

// --- Define Variables we’ll be connecting to ---
double Setpoint, Input, Output;

// --- Specify initial tuning parameters ---
double Kp = 2, Ki = 5, Kd = 1;

// Create PID object: (Input, Output, Setpoint, Kp, Ki, Kd, Direction)
PID myPID(&Input, &Output, &Setpoint, Kp, Ki, Kd, DIRECT);

void setup() {
  Serial.begin(9600);
  Setpoint = 100;              // Target value
  Input = analogRead(PIN_INPUT) * 255 / 1023;
  myPID.SetMode(AUTOMATIC);    // Enable PID
  myPID.SetOutputLimits(0, 255); // Limit PWM output range
}

void loop() {
  Input = analogRead(PIN_INPUT);  // Read process variable
  myPID.Compute();                // Calculate PID output
  analogWrite(PIN_OUTPUT, Output); // Drive actuator

  Serial.print("Input: "); Serial.print(Input);
  Serial.print("\tOutput: "); Serial.println(Output);
  delay(100);
}c:\Users\br425\Downloads\pid_close_loop.ino