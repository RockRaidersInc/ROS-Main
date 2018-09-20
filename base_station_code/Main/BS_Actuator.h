#ifndef __BS_Actuator_H
#define __BS_Actuator_H

#include <SoftwareSerial.h>
#include <Arduino.h>

//PID constants
#define ACTUATOR_KP 10
#define ACTUATOR_KI 0
#define ACTUATOR_KD 0

//Maximum allowable deviation from desired ADC reading
#define ACTUATOR_MAX_ERROR 10

class BS_Actuator{
public:
    BS_Actuator(SoftwareSerial& s, uint8_t pin);
    
    float getElevation() const {return adc_to_angle(position);}
    
    //Reads pot, updates position, does PID, actuates
    void update(const float& desired);

private:
    SoftwareSerial* sabertooth; //pointer to motor controller serial connection
    uint8_t adc_pin;
    int prev_error, integral;   //PID memory
    unsigned int position;  //Latest ADC reading
    
    void drive(int signal); //Actuates
    void read() {position=analogRead(adc_pin);}
    
    //Helper functions to convert between ADC value and angle of elevation
    float adc_to_angle(unsigned int adc) const;
    unsigned int angle_to_adc(float angle) const;

};

#endif