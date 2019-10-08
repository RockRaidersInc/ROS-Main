#!/usr/bin/env python
import rospy
import serial

import serial.tools.list_ports
import time
import pdb
import signal
import os
import random
import threading, sys, traceback

from nmea_msgs.msg import Sentence

# from https://stackoverflow.com/questions/132058/showing-the-stack-trace-from-a-running-python-application
# prints stack traces for all threads
def dumpstacks(signal, frame):
    id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId,""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    return "\n".join(code)


def SIGALARM_handler(signum, frame):
    print "watchdog ran out!"
    stacks = dumpstacks(signum, frame)
    rospy.logerr(str(os.getpid()) + ": serial timeout, call stack written to log")
    print stacks
    raise Exception("end of time")

#sentence: "$GPGGA,223104.024,4243.8223,N,07340.7668,W,1,09,0.9,69.3,M,-32.3,M,,0000*58"
# 2: Latitude
# 3: Latitude direction
# 4: Longitude
# 5: Longitude direction
# 9: Altitude

class BaseStationSerial:

    def __init__(self, dev='/dev/ttyACM0'):
        rospy.Subscriber('/gps/nmea_sentence', Sentence, self.callback_sentence)
        #rospy.Subscriber('/sensor/nmea_sentences', Sentence, self.callback_sentence)

        self.ser = serial.Serial(dev, 9600, timeout=1)

        self.gps_update = False

        while not rospy.is_shutdown():
            signal.setitimer(signal.ITIMER_REAL, 0)  # set the watchdog for 0.25 seconds
            while self.ser.in_waiting:  # Or: while ser.inWaiting():
                print(self.ser.readline())
            signal.setitimer(signal.ITIMER_REAL, .5)  # set the watchdog for 0.25 seconds
            #self.ser.reset_input_buffer()
            try:
                if self.gps_update:
                    #message = 'x,' + str(self.latitude) + ',' + str(self.longitude) + ',' + str(self.altitude) + ','
                    message = 'x,' + str(self.longitude) + ',' + str(self.latitude) + ',' + str(self.altitude) + ','
                    print(message)
                    #print('Ser started')
                    self.ser.write(message)
                    #print('Ser finished')
                    self.gps_update = False
            except Exception as e:
                rospy.logerr("SIGLARAM")

        self.ser.close()

    def callback_sentence(self, msg):
        values = msg.sentence.split(',')
        if values[0] == '$GPGGA':
            #print('Got Message')
            # Correct message type
            self.latitude = float(values[2]) if values[3] == 'N' else -1*float(values[2])
            self.longitude = float(values[4]) if values[5] == 'E' else -1*float(values[4])
            self.altitude = float(values[9])

            self.gps_update = True

        

if __name__ == '__main__':
    signal.signal(signal.SIGALRM, SIGALARM_handler)

    rospy.init_node('baseStationSerial', anonymous=True)
    baseStationSerial = BaseStationSerial(dev='/dev/ttyACM0')
