#ifndef __BS_GPS_H
#define __BS_GPS_H

#include <SoftwareSerial.h>

#define DEFAULT_RX 10
#define DEFAULT_TX 11

class BS_GPS{
public:
	BS_GPS();
	BS_GPS(unsigned int rx, unsigned int tx);
	
    //Needs to be called in setup in order to work;
	void begin(){serial->begin(9600);}
	
	const float& getLongitude() const{return lon;}
	const float& getLatitude() const{return lat;}
	const float& getAltitude() const{return alt;}
	const uint16_t& getCounter() const{return counter;}
	
	bool read();

private:
	SoftwareSerial* serial; //Pointer to serial connection
    //Memory for read function
	bool reading;
	uint8_t comma_counter, state;
	uint16_t counter;   //Number of readings taken
	String output;
    
	float lon, lat, alt, temp_lon, temp_lat, temp_alt;
	
	
};

#endif