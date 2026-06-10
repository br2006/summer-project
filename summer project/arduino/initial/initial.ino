int directionPin = 12;
int pwmPin = 3;
int brakePin = 9;

float t = millis();
float dt = 0;
int potin = A0;

//uncomment if using channel B, and remove above definitions
//int directionPin = 13;
//int pwmPin = 11;
//int brakePin = 8;

void setup() {

  //define pins
  pinMode(directionPin, OUTPUT);
  pinMode(pwmPin, OUTPUT);
  pinMode(brakePin, OUTPUT);
  Serial.begin(9600);

}


void loop() {


  dt = millis() - t;
  t = millis();
  Serial.println(analogRead(A2));



  //release breaks
  digitalWrite(brakePin, LOW);

  //set work duty for the motor
  analogWrite(pwmPin, 255);

  // delay(1000);

  // activate breaks
  // digitalWrite(brakePin, HIGH);

  // delay(1000);

  // set work duty for the motor to 0 (off)
  // analogWrite(pwmPin, 0);

  
}