int motorPin = 3; 
char currentMode = '0'; 

void setup() {
  pinMode(motorPin, OUTPUT);
  Serial.begin(9600);
  Serial.println("--- Continuous Control Ready ---");
  Serial.println("Patterns: H (Heart), S (Shudder), R (Ramp)");
  Serial.println("Noise: N (Heavy Noise), X (Spark/Crackle)");
  Serial.println("Constant: 1, 2, 3 | Stop: 0");
}

void loop() {
  // 1. Check for a NEW command
  if (Serial.available() > 0) {
    char temp = Serial.read();
    
    // Ignore formatting characters
    if (temp != '\n' && temp != '\r' && temp != ' ') {
      currentMode = temp;
      Serial.print("Mode Locked: ");
      Serial.println(currentMode);
      
      if (currentMode == '0') {
        analogWrite(motorPin, 0);
      }
    }
  }

  // 2. Keep running the locked mode
  if (currentMode == '1') { analogWrite(motorPin, 100); }
  else if (currentMode == '2') { analogWrite(motorPin, 180); }
  else if (currentMode == '3') { analogWrite(motorPin, 255); }
  else if (currentMode == 'H') { heartbeat(); }
  else if (currentMode == 'S') { shudder(); }
  else if (currentMode == 'R') { rampUp(); }
  else if (currentMode == 'N') { heavyNoise(); }
  else if (currentMode == 'X') { sparkNoise(); }
}

// --- RHYTHM & NOISE LIBRARY ---

void heartbeat() {
  analogWrite(motorPin, 255); delay(150); 
  analogWrite(motorPin, 0);   delay(150);
  analogWrite(motorPin, 255); delay(400); 
  analogWrite(motorPin, 0);   delay(600); 
}

void shudder() {
  analogWrite(motorPin, 200); delay(40); 
  analogWrite(motorPin, 0);   delay(40);
}

void rampUp() {
  for(int i=0; i<=255; i+=5) { 
    analogWrite(motorPin, i); delay(15);
    if (Serial.available()) return; 
  }
  for(int i=255; i>=0; i-=5) { 
    analogWrite(motorPin, i); delay(15);
    if (Serial.available()) return;
  }
}

// Low frequency pulses with intense, "gritty" internal noise
void heavyNoise() {
  // The "Thump" is made of 4 tiny, random micro-stutters
  for(int i = 0; i < 4; i++) {
    analogWrite(motorPin, 255);      // Full intensity
    delay(random(10, 40));           // Random "on" time creates noise
    analogWrite(motorPin, random(100, 200)); // Don't turn off, just stumble
    delay(random(100, 200));
  }
  analogWrite(motorPin, 0);          // Turn off
  delay(100);                       // Long gap for low-frequency rhythm
}

// High intensity "sparking" or "crackling"
void sparkNoise() {
  analogWrite(motorPin, random(180, 255)); // Irregular intensity
  delay(random(5, 25));                    // Rapid-fire frequency
  if (random(0, 10) > 7) {                 // Random "dead" spots
    analogWrite(motorPin, 0);
    delay(random(10, 50));
  }
}
