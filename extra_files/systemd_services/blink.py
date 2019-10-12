#!/usr/bin/env python

"""
This file is part of the system that blinks the status light when autonomy is active. 
All this file does is tell the arduino to blink the light for a few seconds, it gets run
every two seconds by blinking.sh.
"""

import serial
import time

ser = serial.Serial('/dev/serial/by-id/usb-Arduino_Srl_Arduino_Mega_7563331313335180E041-if00', 9600, timeout=1)

ser.write('a')

#time.sleep(0.1)
#ser.write('a')

