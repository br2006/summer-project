#include <SimpleFOC.h>
// software interrupt library
#include <PciManager.h>
#include <PciListenerImp.h>


// BLDC motor init
BLDCMotor motor = BLDCMotor(11);
// define BLDC driver
BLDCDriver3PWM driver = BLDCDriver3PWM(9, 10, 11, 8);
//Motor encoder init
Encoder encoder = Encoder(2, 3, 500);
// interrupt routine 
void doA(){encoder.handleA();}
void doB(){encoder.handleB();}


// pendulum encoder init
Encoder pendulum = Encoder(A1, A2, 1000);
// interrupt routine 
void doPA(){pendulum.handleA();}
void doPB(){pendulum.handleB();}
// PCI manager interrupt
PciListenerImp listenerPA(pendulum.pinA, doPA);
PciListenerImp listenerPB(pendulum.pinB, doPB);

void setup() {
  
  // initialize motor encoder hardware
  encoder.init();
  encoder.enableInterrupts(doA,doB);
  
  // driver config
  driver.voltage_power_supply = 12;
  driver.init();
  
  // init the pendulum encoder
  pendulum.init();
  PciManager.registerListener(&listenerPA);
  PciManager.registerListener(&listenerPB);
  
  // set control loop type to be used
  motor.torque_controller = TorqueControlType::voltage;
  motor.controller = MotionControlType::torque;

  // link the motor to the encoder
  motor.linkSensor(&encoder);
  // link the motor to the driver
  motor.linkDriver(&driver);
  
  // initialize motor
  motor.init();
  // align encoder and start FOC
  motor.initFOC();
  
}

// loop down-sampling counter
long loop_count = 0;

void loop() {
  // ~1ms 
  motor.loopFOC();

  // pendulum sensor read
  pendulum.update();

  // control loop each ~25ms
  if(loop_count++ > 25){
    
    // calculate the pendulum angle 
    float pendulum_angle = constrainAngle(pendulum.getAngle() + _PI);

    float target_voltage;
    if( abs(pendulum_angle) < 0.5 ) // if angle small enough stabilize
      target_voltage = controllerLQR(pendulum_angle, pendulum.getVelocity(), motor.shaft_velocity);
    else // else do swing-up
      // sets 40% of the maximal voltage to the motor in order to swing up
      target_voltage = -_sign(pendulum.getVelocity())*driver.voltage_power_supply*0.4;

    // set the target voltage to the motor
    motor.move(target_voltage);

    // restart the counter
    loop_count=0;
  }
   

}

// function constraining the angle in between -pi and pi, in degrees -180 and 180
float constrainAngle(float x){
    x = fmod(x + _PI, _2PI);
    if (x < 0)
        x += _2PI;
    return x - _PI;
}

// LQR stabilization controller functions
// calculating the voltage that needs to be set to the motor in order to stabilize the pendulum
float controllerLQR(float p_angle, float p_vel, float m_vel){
  // if angle controllable
  // calculate the control law 
  // LQR controller u = k*x
  //  - k = [40, 7, 0.3]
  //  - x = [pendulum angle, pendulum velocity, motor velocity]' 
  float u =  40*p_angle + 7*p_vel + 0.3*m_vel;
  
  // limit the voltage set to the motor
  if(abs(u) > driver.voltage_power_supply*0.7) u = sign(u)*driver.voltage_power_supply*0.7;
  
  return u;
}
