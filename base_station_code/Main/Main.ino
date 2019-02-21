//---------------------------------
//Active Tracking Base Station Code
//
//Author: Connor McGowan
//---------------------------------


//Serial Controls
//"w" Tilt antenna one step up
//"s" Tilt antenna one step down
//"a" Rotate antenna one step counterclockwise
//"d" Rotate antenna one step clockwise
//"x,latitude,longitude,altitude," Rotate to point at given position
//Format for autonomy:
//There MUST be a comma at the end
//No spaces
//latitude, longitude, and altitude are float and MUST contain a decimal point

#include "BS_GPS.h"
#include "BS_Mag.h"
#include "BS_Motor.h"
#include "BS_Actuator.h"

//The analog pin that the actuator's potentiometer is connected to
#define ACTUATOR_ADC_PIN A0

//Size in degrees of one manual step of elevation
#define ACTUATOR_MANUAL_STEP 1
//Size in degrees of one manual step of azimuth
#define MOTOR_MANUAL_STEP 15

//Flag and timer for reading magnetometer
bool mag_flag=false;

//Variables for holding rover telemetry
float rover_lat=0;
float rover_lon=0;
float rover_alt=0;

float desired_heading=0;
float desired_elevation=0;

char command='i';

BS_GPS gps;
BS_Mag mag;
//THE SABERTOOTH MUST BE INITIALIZED AFTER THE GPS
SoftwareSerial Sabertooth(4, 13);
BS_Motor motor(Sabertooth);
BS_Actuator actuator(Sabertooth, ACTUATOR_ADC_PIN);

void setup() {
  //Begin all communication
  Serial.begin(9600);   //THIS IS ONLY HERE WHILE SERIAL IS BEING USED
  Sabertooth.begin(9600);
  gps.begin();
  mag.init();

  //Initialize motors to both be stopped
  Sabertooth.write((byte)0);
  
  //Set up the magnetometer and timer
  timer_init();

  //Take 1 second of readings to determine initial heading
  float heading_sum=0;
  for(unsigned char i=0; i<20; i++){
    while(!mag.read(mag_flag));
    heading_sum+=mag.getHeading();
  }
  motor.init(heading_sum/20); //Give initial heading to motor object

  //Get 10 GPS readings to intialize position
  while(gps.getCounter()<10)
    gps.read();

  
}

void loop() {
  // put your main code here, to run repeatedly:

  //Update GPS if possible
  gps.read();

  receiveData();  //THIS IS ONLY HERE WHILE SERIAL IS BEING USED

  //Do all measurements and calculations every 50ms
  if(mag.read(mag_flag)){
    //update actuator reading
    actuator.read();

    switch(command){
      case 'e': //autonomous mode
        desired_heading=-1*get_bearing();
        desired_elevation=atan((rover_alt-gps.getAltitude())/get_distance())*180/PI;
        break;
        
      case 'w': //move up one step
        desired_heading=mag.getHeading();
        desired_elevation=actuator.getElevation()+ACTUATOR_MANUAL_STEP;
        command='i';
        break;

      case 's': //move down one step
        desired_heading=mag.getHeading();
        desired_elevation=actuator.getElevation()-ACTUATOR_MANUAL_STEP;
        command='i';
        break;

      case 'a': //move counterclockwise one step
        desired_heading=mag.getHeading()+MOTOR_MANUAL_STEP;
        if(desired_heading>180)
          desired_heading-=360;
        desired_elevation=actuator.getElevation();
        command='i';
        break;

      case 'd': //move clockwise one step
        desired_heading=mag.getHeading()-MOTOR_MANUAL_STEP;
        if(desired_heading<-180)
          desired_heading+=360;
        desired_elevation=actuator.getElevation();
        command='i';
        break;
    }

    
    motor.update(mag.getHeading(),desired_heading);
    actuator.update(desired_elevation);
  }
}

//Initialize timer to tick at 220Hz
void timer_init(){
  TCCR1A = 0;
  TCCR1B = 0x0A; //Sets CTC mode, prescaler of 8
  TIMSK1 |= 0x02; // enable timer compare interrupt
  TCNT1  = 0; //clear timer
  OCR1A = 9091; // compare match register 16MHz/8/220Hz
  
}

//ISR for timer
//Sets time-based variables for use elsewhere
ISR(TIMER1_COMPA_vect){
  interrupts(); //turn on interrupts so I2C still works
  mag_flag = true;  //ready to read magnetometer
}

//Calculates bearing from latitude and longitude of base station and target
//returns bearing between -180 and 180
//North is 0, West is negtive, East is positive (opposite convention)
float get_bearing(){
  //Convert coordinates to radians
  float lat_rad = gps.getLatitude()*PI/180;
  float lon_rad = gps.getLongitude()*PI/180;
  float rover_lat_rad = rover_lat*PI/180;
  float rover_lon_rad = rover_lon*PI/180;
  return atan2(sin(rover_lon_rad-lon_rad)*cos(rover_lat_rad),cos(lat_rad)*sin(rover_lat_rad)-sin(lat_rad)*cos(rover_lat_rad)*cos(rover_lon_rad-lon_rad))*180/PI;
}

//Calculates distance between base station and target
//Uses Haversine formula
float get_distance(){
  //Convert coordinates to radians
  float lat_rad = gps.getLatitude()*PI/180;
  float lon_rad = gps.getLongitude()*PI/180;
  float rover_lat_rad = rover_lat*PI/180;
  float rover_lon_rad = rover_lon*PI/180;
  float a = pow(sin((rover_lat_rad-lat_rad)/2),2)+cos(rover_lat_rad)*cos(lat_rad)*pow(sin((rover_lon_rad-lon_rad)/2),2);
  return 2*atan2(sqrt(a), sqrt(1-a))*6371000;
}

void receiveData(){ //THIS IS ONLY HERE WHILE SERIAL IS BEING USED
  char input=';';
  int decimal_places=0;
  if(Serial.available()){
    command=Serial.read();
    if(command=='x'){
      rover_lat=0;
      while(!Serial.available());
      Serial.read();
      while(!Serial.available());
      input=Serial.read();
      while(input!='.'){
        rover_lat=rover_lat*10+input-'0';
        while(!Serial.available());
        input=Serial.read();
      }
      while(!Serial.available());
      input=Serial.read();
      while(input!=','){
        decimal_places++;
        rover_lat+=(float)(input-'0')/pow(10, decimal_places);
        while(!Serial.available());
        input=Serial.read();
      }
      decimal_places=0;
      rover_lon=0;
      while(!Serial.available());
      input=Serial.read();
      while(input!='.'){
        rover_lon=rover_lon*10+input-'0';
        while(!Serial.available());
        input=Serial.read();
      }
      while(!Serial.available());
      input=Serial.read();
      while(input!=','){
        decimal_places++;
        rover_lon+=(float)(input-'0')/pow(10, decimal_places);
        while(!Serial.available());
        input=Serial.read();
      }
      decimal_places=0;
      rover_alt=0;
      while(!Serial.available());
      input=Serial.read();
      while(input!='.'){
        rover_alt=rover_alt*10+input-'0';
        while(!Serial.available());
        input=Serial.read();
      }
      while(!Serial.available());
      input=Serial.read();
      while(input!=','){
        decimal_places++;
        rover_alt+=(float)(input-'0')/pow(10, decimal_places);
        while(!Serial.available());
        input=Serial.read();
      }
      decimal_places=0;
    }
  }
}

