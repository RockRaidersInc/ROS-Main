#include "BS_GPS.h"
#include <math.h>

BS_GPS::BS_GPS(){
	serial= new SoftwareSerial(DEFAULT_RX, DEFAULT_TX);
	reading=false;
	comma_counter=0;
	state=0;
	counter=0;
	output="";
	lon=lat=alt=temp_lon=temp_lat=temp_alt=0;
}

BS_GPS::BS_GPS(unsigned int rx, unsigned int tx){
	serial=new SoftwareSerial(rx, tx);
	reading=false;
	comma_counter=0;
	state=0;
	counter=0;
	output="";
	lon=lat=alt=temp_lon=temp_lat=temp_alt=0;
}

bool BS_GPS::read(){
	//Get next character if available
	if(serial->available()){
		char input = serial->read();

		//Check if it's the beginning of a NMEA sentence
		if(input=='$'){
			reading=true; //flag to store next part of sentence
			//Reset variables
			comma_counter=0;
			output="";
			state=1;
			return false; //wait for next character
		}

		//For parsing through useless parts
		//Skips until a certain number of commas have passed
		if(comma_counter>0){
			if(input==',')
				comma_counter--;
			return false; //wait for next character
		}

    
		if(reading){  //read into output until next comma
			if(input==',')  //We're done reading the part we care about
				reading=false;
			else{
				output+=input;  //Store this part in output
				return false; //wait for next character
			}
		}

    
		switch(state){
			default:  //state is 0, don't care about this stuff
				break;
			
			case 1: //checking sentence type
				if(output.equals("GPGGA")){
					comma_counter=1;  //skip next field
					reading=true;
					state=2;  //advance state
				}
				else  //go back to not caring
					state=0;
				output=""; //clear reading
				break;
			
			case 2: //reading latitude
				temp_lat = output.toFloat();
				//convert from degrees, minutes to decimal degrees
				temp_lat = floor(temp_lat/100)+(temp_lat/100-floor(temp_lat/100))/.6;
				output="";  //clear reading
				state=3; //advance state
				break;
			
			case 3: //check direction of latitude
				if(input=='S')
					temp_lat*=-1;
				state=4; //advance state
				comma_counter=1;  //skip next field
				reading=true;
				break;
			
			case 4: //reading longitude
				temp_lon = output.toFloat();
				//convert from degrees, minutes to decimal degrees
				temp_lon = floor(temp_lon/100)+(temp_lon/100-floor(temp_lon/100))/.6;
				output=""; //clear reading
				state=5; //advance state
				break;
			
			case 5: //check direction of longitude
				if(input=='W')
					temp_lon*=-1;
				state=6; //advance state
				comma_counter=4; //skip 4 fields
				reading=true;
				break;
			
			case 6: //reading altitude
				temp_alt = output.toFloat();
				output=""; //clear reading
				state=0; //return to default state

				//check to prevent overflow of counter
				if(counter<65535){
					counter++;
					//Update cumulative avaerage of readings
					lon = (lon*(counter-1)+temp_lon)/counter;
					lat = (lat*(counter-1)+temp_lat)/counter;
					alt = (alt*(counter-1)+temp_alt)/counter;
				}
				return true;
		}
	}
	return false;
}