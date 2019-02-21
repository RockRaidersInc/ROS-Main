#ifndef __BS_Motor_H
#define __BS_Motor_H

#include <SoftwareSerial.h>

//PID constants
#define MOTOR_KP 36
#define MOTOR_KI 0
#define MOTOR_KD 6

//Maximum allowable deviation from desired ADC reading
#define MOTOR_MAX_ERROR 5

class BS_Motor{
public:
    BS_Motor(SoftwareSerial& s);
    
    //Sets the initial heading used to detect wrapping
    //Must be called after first magnetometer reading has been taken, before moving
    void init(float init_heading){initial_heading=prev_heading=init_heading;}
    
    //Checks for wire wraparound, does PID, actuates motor
    void update(float current, float desired);

private:
    SoftwareSerial* sabertooth; //pointer to motor controller serial connection
    float prev_error, integral, prev_heading, initial_heading;
    int wrap;   //Keeps track of wire wraparound
    
    void drive(int signal); //Actuates motor

};

#endif
