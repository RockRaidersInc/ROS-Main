#ifndef __BS_Mag_H
#define __BS_Mag_H

#include <Wire.h>

//Magnetic North isn't always equal to geographic North
//This number corrects for the deviation
#define DECLINATION 5

//Calibration factors
#define X_GAIN 1.028
#define X_OFFSET 48.5
#define Y_GAIN .972
#define Y_OFFSET 227.5

//The number of readings averaged before updating
#define NUM_READINGS 11

class BS_Mag{
public:
	BS_Mag();
	
    //Adjusts settings, must be called in setup
	void init();
	
	const float& getHeading() const {return heading;}
	
	bool read(bool& flag);

private:
	uint8_t counter;    //Number of readings taken
	float heading;
	long x_sum, y_sum;

};

#endif