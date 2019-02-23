/*
 *  This code reads data from the IMU and writes it to the serial line. It has an associated ROS node named imu_publisher.
 */

#define _SS_MAX_RX_BUFF 256

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_LSM303_U.h>
#include <Adafruit_L3GD20_U.h>
#include <Adafruit_9DOF.h>
#include <SoftwareSerial.h>

#define _SS_MAX_RX_BUFF 256

/* Assign a unique ID to the sensors */
Adafruit_LSM303_Accel_Unified accel = Adafruit_LSM303_Accel_Unified(30301);
Adafruit_LSM303_Mag_Unified   mag   = Adafruit_LSM303_Mag_Unified(30302);
Adafruit_L3GD20_Unified       gyro  = Adafruit_L3GD20_Unified(20);

unsigned long sequence_num;


SoftwareSerial gps_serial(13, 12); // RX, TX


void displaySensorDetails(void)
{
  sensor_t sensor;
  
  accel.getSensor(&sensor);
  Serial.println(F("----------- ACCELEROMETER ----------"));
  Serial.print  (F("Sensor:       ")); Serial.println(sensor.name);
  Serial.print  (F("Driver Ver:   ")); Serial.println(sensor.version);
  Serial.print  (F("Unique ID:    ")); Serial.println(sensor.sensor_id);
  Serial.print  (F("Max Value:    ")); Serial.print(sensor.max_value); Serial.println(F(" m/s^2"));
  Serial.print  (F("Min Value:    ")); Serial.print(sensor.min_value); Serial.println(F(" m/s^2"));
  Serial.print  (F("Resolution:   ")); Serial.print(sensor.resolution); Serial.println(F(" m/s^2"));
  Serial.println(F("------------------------------------"));
  Serial.println(F(""));

  gyro.getSensor(&sensor);
  Serial.println(F("------------- GYROSCOPE -----------"));
  Serial.print  (F("Sensor:       ")); Serial.println(sensor.name);
  Serial.print  (F("Driver Ver:   ")); Serial.println(sensor.version);
  Serial.print  (F("Unique ID:    ")); Serial.println(sensor.sensor_id);
  Serial.print  (F("Max Value:    ")); Serial.print(sensor.max_value); Serial.println(F(" rad/s"));
  Serial.print  (F("Min Value:    ")); Serial.print(sensor.min_value); Serial.println(F(" rad/s"));
  Serial.print  (F("Resolution:   ")); Serial.print(sensor.resolution); Serial.println(F(" rad/s"));
  Serial.println(F("------------------------------------"));
  Serial.println(F(""));
  
  mag.getSensor(&sensor);
  Serial.println(F("----------- MAGNETOMETER -----------"));
  Serial.print  (F("Sensor:       ")); Serial.println(sensor.name);
  Serial.print  (F("Driver Ver:   ")); Serial.println(sensor.version);
  Serial.print  (F("Unique ID:    ")); Serial.println(sensor.sensor_id);
  Serial.print  (F("Max Value:    ")); Serial.print(sensor.max_value); Serial.println(F(" uT"));
  Serial.print  (F("Min Value:    ")); Serial.print(sensor.min_value); Serial.println(F(" uT"));
  Serial.print  (F("Resolution:   ")); Serial.print(sensor.resolution); Serial.println(F(" uT"));  
  Serial.println(F("------------------------------------"));
  Serial.println(F(""));

  delay(500);
}



void print_space_if_positive(float x) {
 if (x >= 0.0) {
    Serial.print(" ");
 } 
 if (abs(x) < 10.0) {
    Serial.print(" "); 
 }
}


void setup(void)
{
//  Serial.begin(115200);
  Serial.begin(19200);
  Serial.println(F("Adafruit 9DOF Tester")); Serial.println("");
  
  /* Initialise the sensors */
  if(!accel.begin())
  {
    /* There was a problem detecting the ADXL345 ... check your connections */
    Serial.println(F("Ooops, no LSM303 detected ... Check your wiring!"));
    while(1);
  }
  if(!mag.begin())
  {
    /* There was a problem detecting the LSM303 ... check your connections */
    Serial.println("Ooops, no LSM303 detected ... Check your wiring!");
    while(1);
  }
  if(!gyro.begin())
  {
    /* There was a problem detecting the L3GD20 ... check your connections */
    Serial.print("Ooops, no L3GD20 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  
  /* Display some basic information on this sensor */
  displaySensorDetails();
  
  Serial.println(F("ACCEL X Y Z (m/s^2), MAG X Y Z (uT), GYRO X Y Z (rad/s)"));
  
  sequence_num = 0;
  
  gps_serial.begin(9600);
}


void check_gps_data() {
  while (gps_serial.available()) {
    Serial.write("!");  
    char next = gps_serial.read();
    Serial.write(next);
  } 
}


void loop(void)
{
  unsigned long loop_start_time = millis();  // this will overflow after about 50 days, it isn't an issue for us
  
  
  /* Get a new sensor event */
  sensors_event_t accel_event;
  sensors_event_t gyro_event;
  sensors_event_t mag_event;

  /* Display the results (acceleration is measured in m/s^2) (magnetic vector values are in micro-Tesla (uT)) (gyrocope values in rad/s) */
  accel.getEvent(&accel_event);
  gyro.getEvent(&gyro_event);
  mag.getEvent(&mag_event);
  check_gps_data();


  Serial.print("Seq:"); Serial.print(sequence_num);
  
  Serial.print(", ms_time:"); Serial.print(loop_start_time);
  Serial.print(", ");
  print_space_if_positive(accel_event.acceleration.x); Serial.print(accel_event.acceleration.x); Serial.print(" ");
  print_space_if_positive(accel_event.acceleration.y); Serial.print(accel_event.acceleration.y); Serial.print(" "); 
  print_space_if_positive(accel_event.acceleration.z); Serial.print(accel_event.acceleration.z); Serial.print(", ");
  check_gps_data();
  

  /* Display the results (magnetic vector values are in micro-Tesla (uT)) */
  print_space_if_positive(mag_event.magnetic.x); Serial.print(mag_event.magnetic.x); Serial.print(" "); 
  print_space_if_positive(mag_event.magnetic.y);  Serial.print(mag_event.magnetic.y); Serial.print(" "); 
  print_space_if_positive(mag_event.magnetic.z); Serial.print(mag_event.magnetic.z); Serial.print(", ");
  check_gps_data();


  /* Display the results (gyrocope values in rad/s) */
  print_space_if_positive(gyro_event.gyro.x); Serial.print(gyro_event.gyro.x); Serial.print(" ");
  print_space_if_positive(gyro_event.gyro.y); Serial.print(gyro_event.gyro.y); Serial.print(" ");
  print_space_if_positive(gyro_event.gyro.z); Serial.print(gyro_event.gyro.z);
  Serial.println("");
  check_gps_data();
  

  // the loop should run at most every 10 ms
  int loop_time = 10;
  while (millis() - loop_start_time < loop_time)
  {
     delay(1); 
  }
  
  sequence_num += 1;
}
