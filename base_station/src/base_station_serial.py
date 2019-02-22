#!/usr/bin/env python
import rospy
import serial

from nmea_msgs.msg import Sentence

#sentence: "$GPGGA,223104.024,4243.8223,N,07340.7668,W,1,09,0.9,69.3,M,-32.3,M,,0000*58"
# 2: Latitude
# 3: Latitude direction
# 4: Longitude
# 5: Longitude direction
# 9: Altitude

class BaseStationSerial:

    def __init__(self, dev='/dev/ttyACM0'):
        rospy.Subscriber('/gps/nmea_sentences', Sentence, self.callback_sentence)

        self.ser = serial.Serial(dev)

        self.gps_update = False

        while not rospy.is_shutdown():
            if self.gps_update:
                message = 'x,' + str(self.latitude) + ',' + str(self.longitude) + ',' + str(self.altitude) + ','
                ser.write(message)
                self.gps_update = False
            rospy.sleep(1)

        ser.close()

    def callback_sentence(self, msg):
        values = msg.sentence.split(',')
        if values[0] == '$GPGGA':
            # Correct message type
            self.longitude = float(values[2]) if values[3] == 'N' else -1*float(values[2])
            self.latitude = float(values[4]) if values[5] == 'E' else -1*float(values[4])
            self.altitude = float(values[9])

            self.gps_update = True

        

if __name__ == '__main__':
    rospy.init_node('baseStationSerial', anonymous=True)
    baseStationSerial = BaseStationSerial(dev='/dev/ttyACM0')
