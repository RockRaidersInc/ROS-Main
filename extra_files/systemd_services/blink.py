#!/usr/bin/env python

import serial
import time

ser = serial.Serial('/dev/serial/by-id/usb-Arduino_Srl_Arduino_Mega_7563331313335180E041-if00', 9600, timeout=1)

ser.write('a')

#time.sleep(0.1)
#ser.write('a')

