#include "BS_Motor.h"

BS_Motor::BS_Motor(SoftwareSerial& s){
    sabertooth=&s;
    prev_error=integral=prev_heading=initial_heading=wrap=0;
}

void BS_Motor::update(float current, float desired){
    //Check if antenna has wrapped around +/- 180 degrees
    if(current > 90 && prev_heading < -90) //crossed from negative to positive
        wrap -= 360;  //cumulative heading is 360 degrees less than true heading
    if(prev_heading > 90 && current < -90) //crossed from positive to negative
        wrap += 360;  //cumulative heading is 360 degrees more than true heading
        
    float error=desired-current;
    //Constrain error between -180 and 180
    if (error > 180) 
        error -= 360;
    if (error < -180) 
        error += 360;
    
    //Check if antenna is trying to wrap around more than 360 degrees from starting position
    //If so, send it 360 degrees in the other direction
    if(wrap+current+error > initial_heading + 360)
        error-=360;
    else if(wrap+current+error < initial_heading - 360)
        error+=360;
    
    //Do PID
    integral+=error;
    int pid = MOTOR_KP*error+MOTOR_KI*integral+MOTOR_KD*(error-prev_error);
    prev_error=error;
    
    //Actuate
    if(abs(error)<=MOTOR_MAX_ERROR)
        drive(0);
    else
        drive(pid);
}

void BS_Motor::drive(int signal){
    if(signal<0){
        signal*=-1;
        if(signal > 63) //limit to maximum value that can be written
            signal = 63;
        sabertooth->write(192-signal); //Speeds for motor are centered around 192
    }
    else{
        if(signal > 63) //limit to maximum value that can be written
            signal = 63;
        sabertooth->write(192+signal); //Speeds for motor are centered around 192
    }
}