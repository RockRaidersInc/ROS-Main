/**
 * DAC for the Arm Encoder - xieo
 * Outputs 0-2v to represent reotational position.
 */

#include <Wire.h> // For I2C

#define MCP4725_ADDR 0x60

int vddPin = 2; // VDD for the encoder
int csPin = 3; // Chip select pin - output to request bits
int clkPin = 4; // Serial Clock pin - switch to indicate next bit
int doPin = 5; // Serial Data output pin - the current bit

int maxPosVal = 1023; // Max digital position
int maxOutVal = 4095; // Max digital to DAC val (5v)

void setup() {
  pinMode(vddPin, OUTPUT);
  pinMode(csPin, OUTPUT);
  pinMode(clkPin, OUTPUT);
  pinMode(doPin, INPUT);

  // Set vcc to encoder to 5v
  digitalWrite(vddPin, HIGH);

  // Set Chip Select and Clock to high - block input
  digitalWrite(csPin, HIGH);
  digitalWrite(clkPin, HIGH);

  Wire.begin(); // For DAC
  Serial.begin(115200);
}

void loop() {
  int val = 0; // Current positional value of the encoder
  int out = 0; // Current output value to the DAC
  int i = 0; // Index of the bin (0 = least significant bit, first bit)
  
  // Activate Data Output and wait 1 microseconds (min 100 ns)
  digitalWrite(csPin, LOW);
  delayMicroseconds(1);

  // Read in the 10 bits
  for(i = 0; i < 10; i++) {
    // Clock falling edge, wait for return for data ready to be output
    digitalWrite(clkPin, LOW);
    delayMicroseconds(1); // Wait 1 microseconds (min 500 ns)
    
    // Clock back to High, wait until data output is valid
    digitalWrite(clkPin, HIGH);
    delayMicroseconds(1); // Wait 1 micrseconds (min 375 ns)
    
    // Read pin
    boolean bitValue = digitalRead(doPin);
    val = (val << 1) | bitValue; 
  }
  // Reset loop, wait 1 micrcroseconds for tristate (min 100 ns)
  digitalWrite(csPin, HIGH);

  // Map scale 1024 to 4096;
  out = val << 2;
  out = (int) ((out << 1) / 3.3);
  Serial.println(out);

  Wire.beginTransmission(MCP4725_ADDR);
  Wire.write(64);                     // cmd to update the DAC
  Wire.write(out >> 4);        // the 8 most significant bits...
  Wire.write((out & 15) << 4); // the 4 least significant bits...
  Wire.endTransmission();

  delayMicroseconds(1);
}
