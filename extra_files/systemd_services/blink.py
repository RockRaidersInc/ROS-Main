#!/usr/bin/env python

import serial
import time

ser = serial.Serial('/dev/serial/by-id/usb-Arduino_Srl_Arduino_Mega_7563331313335180E041-if00', 9600, timeout=1)

ser.write('a')

print "hi"

time.sleep(1.0)
ser.write('a')



