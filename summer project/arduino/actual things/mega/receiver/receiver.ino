
void setup() {
  Serial.begin(9600);    // USB monitor
  Serial1.begin(9600);   // RX1=19, TX1=18
  Serial.println("MEGA listening on Serial1...");
}

void loop() {
  while (Serial1.available()) {
    char c = Serial1.read();
    Serial.write(c);     // show incoming on monitor
  }
}