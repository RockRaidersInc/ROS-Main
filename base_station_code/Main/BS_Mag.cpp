#include "BS_Mag.h"
#include <math.h>

#define PI 3.14159265359

BS_Mag::BS_Mag(){
	counter=0;
	heading=0;
	x_sum=y_sum=0;
}

void BS_Mag::init(){
	Wire.begin();
	
	Wire.beginTransmission(0x1E);
	Wire.write(0x00); //CRA_REG_M
	Wire.write(0x1C); //Sets data rate to 220HZ
	Wire.endTransmission();

	Wire.beginTransmission(0x1E);
	Wire.write(0x01); //CRB_REG_M
	Wire.write(0x20); //Sets range to +/- 1.3 Gauss
	Wire.endTransmission();

	Wire.beginTransmission(0x1E);
	Wire.write(0x02); //MR_REG_M
	Wire.write(0x00); //Continuous-conversion mode
	Wire.endTransmission();
}

//The flag reference should be set by a timer at the measurement frequency
//Returns true when a new data point is ready
bool BS_Mag::read(bool& flag){
    //Don't take a reading if the flag hasn't triggered
	if(!flag)
		return false;
	flag=false; //reset flag
	int xmag, ymag;
	Wire.beginTransmission(0x1E);
	Wire.write(0x03); //first data register
	Wire.endTransmission();
	Wire.requestFrom(0x1E, 6); //need to get all 6 registers or it won't respond
	while(Wire.available()<6);
	xmag = (Wire.read()<<8)|Wire.read();  //Shift high register, add low register to form reading
	//Throw away z readings, don't need them
	Wire.read();
	Wire.read();
	ymag = (Wire.read()<<8)|Wire.read(); //Shift high register, add low register to form reading
	//Apply calibration settings
	x_sum += (((float)xmag-X_OFFSET/11)*X_GAIN);
	y_sum += (((float)ymag-Y_OFFSET/11)*Y_GAIN);
	counter++;
	
    //When we're ready to average...
	if(counter>=NUM_READINGS){
		heading=atan2(y_sum, -1*x_sum)*180/PI+DECLINATION;
        //Constrain heading between -180 and 180
		if(heading<-180)
			heading+=360;
		if(heading>180)
			heading-=360;
        
		counter=0;
		x_sum=0;
		y_sum=0;
		return true;
	}
	
	return false;
	
}