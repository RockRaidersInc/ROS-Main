#!/usr/bin/env python

import rospy
import roboclaw
from std_msgs.msg import Int8
from std_msgs.msg import Int16

import serial.tools.list_ports
import time


class motornode:
    TIMEOUT_TIME = 1000

    name = ''
    address = 0x80
    device = ''

    timeout = 0

    m1_enc_pub = None
    m2_enc_pub = None

    connected = False

    m1_pwm = None
    m2_pwm = None

    m1_vel = None
    m2_vel = None


    def __init__(self, name, m1_name='M1', m2_name='M2', publish_enc=False, address=0x80):
        self.name = name
        self.address = address
        self.timeout = int(round(time.time() * 1000))

        #PWM from 0 to 127
        #rospy.Subscriber(m1_name + '_pwm', Int8, self.callbackM1_pwm)
        #rospy.Subscriber(m2_name + '_pwm', Int8, self.callbackM2_pwm)

        #Signed velocity in encoder ticks/second
        rospy.Subscriber(m1_name + '_vel', Int16, self.callbackM1_vel)
        rospy.Subscriber(m2_name + '_vel', Int16, self.callbackM2_vel)

        #TODO: Need accel, speed, decel, and pos
        #rospy.Subscriber(m1_name + '_pos', Int8, self.callbackM1_pos)
        #rospy.Subscriber(m2_name + '_pos', Int8, self.callbackM2_pos)

        if publish_enc:
            self.m1_enc_pub = rospy.Publisher(m1_name + '_enc', Int8, queue_size = 1)
            self.m2_enc_pub = rospy.Publisher(m2_name + '_enc', Int8, queue_size = 1)

        while not rospy.is_shutdown():
            # Try to connect every second
            self.connected = self.connect()
            if self.connected:
                rospy.loginfo('%s has connected', self.address)
                #print("connected")
                break
            rospy.sleep(1.0)

        while not rospy.is_shutdown():
            # Publish encoder readings
            if self.m1_enc_pub is not None:
                self.m1_enc_pub.pub(roboclaw.ReadEncM1(self.address))
                self.m2_enc_pub.pub(roboclaw.ReadEncM2(self.address))

            # Timeout if no command recieved for more than TIMEOUT_TIME
            if int(round(time.time() * 1000)) - self.timeout > self.TIMEOUT_TIME:
                roboclaw.ForwardBackwardM1(self.address, 64)
                roboclaw.ForwardBackwardM2(self.address, 64)
            else:
                if self.m1_pwm is not None:
                    roboclaw.ForwardBackwardM1(self.address, self.m1_pwm)
                    self.m1_pwm = None
                if self.m2_pwm is not None:
                    roboclaw.ForwardBackwardM2(self.address, self.m2_pwm)
                    self.m2_pwm = None
                if self.m1_vel is not None:
                    roboclaw.SpeedM1(self.address, self.m1_vel)
                    self.m1_vel = None
                if self.m2_vel is not None:
                    roboclaw.SpeedM2(self.address, self.m2_vel)
                    self.m2_vel = None


    def connect(self):
        ports = serial.tools.list_ports.comports()
        for usb in ports:
            try:
                # Open up the serial port and see if data is available at the desired address
                roboclaw.Open(usb.device, 38400)
                c1, c2 = roboclaw.GetConfig(self.address)
                print(c1)
                print(c2)
                print(' ')
                if (c1 is not 0) or (c2 is not 0):
                    self.device = usb.device
                    rospy.loginfo('%s has connected to roboclaw with address %s', self.device, self.address)
                    #print(self.name + ' ' + self.device)
                    return True
                else:
                    roboclaw.port.close()
            except IOError:
                continue
        return False


    def callbackM1_pwm(self, msg):
        if self.connected:
            self.timeout = int(round(time.time() * 1000))
            self.m1_pwm = msg.data
        else:
            #rospy.loginfo('%s recieved M1_pwm, not connected', self.address)
            pass
          
    def callbackM2_pwm(self, msg):
        if self.connected:
            self.timeout = int(round(time.time() * 1000))
            self.m2_pwm = msg.data
        else:
            #rospy.loginfo('%s recieved M2_pwm, not connected', self.address)
            pass


    def callbackM1_vel(self, msg):
        if self.connected:
            self.timeout = int(round(time.time() * 1000))
            self.m1_vel = msg.data
        else:
            rospy.loginfo('%s recieved M1_vel, not connected', self.address)
            pass

    def callbackM2_vel(self, msg):
        if self.connected:
            self.timeout = int(round(time.time() * 1000))
            self.m2_vel = msg.data
        else:
            rospy.loginfo('%s recieved M2_vel, not connected', self.address)
            pass

    def callbackM1_pos(self, msg):
        pass
    def callbackM2_pos(self, msg):
        pass


if __name__ == '__main__':
    rospy.init_node('motornode', anonymous=True)
    name = rospy.get_param('~controller_name', 'motornode')
    m1Name = rospy.get_param('~m1_name', 'M1')
    m2Name = rospy.get_param('~m2_name', 'M2')
    pub_enc = rospy.get_param('~pub_enc', False)
    address = int(rospy.get_param('~address', 0x80))
    controller = motornode(name, m1Name, m2Name, pub_enc, address)
