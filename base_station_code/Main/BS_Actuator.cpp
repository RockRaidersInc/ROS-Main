#include "BS_Actuator.h"

BS_Actuator::BS_Actuator(SoftwareSerial& s, uint8_t pin){
    sabertooth=&s;
    adc_pin=pin;
    position=integral=prev_error=0;
}

void BS_Actuator::update(float desired){
    if(desired>15)
      desired=15;
    else if(desired<-15)
      desired=-15;
    //Calculate error
    int error=angle_to_adc(desired)-position;
    //Do PID
    integral+=error;
    int pid = ACTUATOR_KP*error+ACTUATOR_KI*integral+ACTUATOR_KD*(error-prev_error);
    prev_error=error;
    //Actuate
    if(abs(error)<=ACTUATOR_MAX_ERROR)
        drive(0);
    else
        drive(pid);
}

void BS_Actuator::drive(int signal){
    if(signal<0){
        signal*=-1;
        if(signal > 63) //limit to maximum value that can be written
            signal = 63;
        sabertooth->write((byte)(64+signal)); //Speeds for actuator are centered around 64
    }
    else{
        if(signal > 63) //limit to maximum value that can be written
            signal = 63;
        sabertooth->write((byte)(64-signal)); //Speeds for actuator are centered around 64
    }
}

float BS_Actuator::adc_to_angle(unsigned int adc) const{
    return (acos((pow((float)(adc+1439)/93.249, 2)-1489.28)/-1293.28)-.609)*180/PI+.5;
}

unsigned int BS_Actuator::angle_to_adc(float angle) const{
    return 93.249*sqrt(1489.28-1293.28*cos((angle-.5)*PI/180+.609))-1439;
}
