#!/usr/bin/env python

import rospy
import roboclaw
from std_msgs.msg import Int8
from std_msgs.msg import Int16
from std_msgs.msg import Int32

import serial.tools.list_ports
import time
import pdb
import signal
import os
import random
import threading, sys, traceback


def SIGALARM_handler(signum, frame):
    print "watchdog ran out!"
    stacks = dumpstacks(signum, frame)
    rospy.logerr(str(os.getpid()) + ": serial timeout, call stack written to log")
    print stacks
    raise Exception("end of time")


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


class motornode:
    TIMEOUT_TIME = 1000

    QPPS = 500

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

    m1_pos = None
    m2_pos = None


    def __init__(self, name, m1_name='M1', m2_name='M2', publish_enc=False, address=0x80):
        self.name = name
        self.address = address
        self.timeout = int(round(time.time() * 1000))

        #PWM from 0 to 127
        rospy.Subscriber(m1_name + '_pwm', Int8, self.callbackM1_pwm)
        rospy.Subscriber(m2_name + '_pwm', Int8, self.callbackM2_pwm)

        #Signed velocity in encoder ticks/second
        rospy.Subscriber(m1_name + '_vel', Int16, self.callbackM1_vel)
        rospy.Subscriber(m2_name + '_vel', Int16, self.callbackM2_vel)

        rospy.Subscriber(m1_name + '_pos', Int32, self.callbackM1_pos)
        rospy.Subscriber(m2_name + '_pos', Int32, self.callbackM2_pos)

        if publish_enc:
            self.m1_enc_pub = rospy.Publisher(m1_name + '_enc', Int32, queue_size = 1)
            self.m2_enc_pub = rospy.Publisher(m2_name + ("_enc2" if m1_name == m2_name else "_enc"), Int32, queue_size = 1)

        while not rospy.is_shutdown():
            # Try to connect every second
            self.connected = self.connect()
            if self.connected:
                rospy.loginfo('%s has connected (' + str((self.name, os.getpid())) + ')', self.address)
                #print("connected")
                break
            rospy.sleep(1.0)

        update_rate = 0.05
        while not rospy.is_shutdown():
            start_time = time.time()
            self.update()
            time.sleep(max(start_time + update_rate - time.time(), 0))

    def update(self):
        signal.setitimer(signal.ITIMER_REAL, 0.25)  # set the watchdog for 0.25 seconds
        try:
            # Publish encoder readings
            if self.m1_enc_pub is not None:
                response_1 = roboclaw.ReadEncM1(self.address)
                response_2 = roboclaw.ReadEncM2(self.address)

                if response_1[0] == 0 or response_1[0] == 0:
                    rospy.logerr(str((self.name, os.getpid())) + ": error returned from encoder reading: " + str(response_1) + " " + str(response_2))
                else:
                    m1_msg = Int32()
                    m1_msg.data = int(response_1[1])
                    m2_msg = Int32()
                    m2_msg.data = int(response_2[1])
                    self.m1_enc_pub.publish(m1_msg)
                    self.m2_enc_pub.publish(m2_msg)

            # Timeout if no command recieved for more than TIMEOUT_TIME
            if int(round(time.time() * 1000)) - self.timeout > self.TIMEOUT_TIME:
                roboclaw.ForwardBackwardM1(self.address, 64)
                roboclaw.ForwardBackwardM2(self.address, 64)
            else:
                if self.m1_pwm is not None:
                    roboclaw.ForwardBackwardM1(self.address, self.m1_pwm)
                    # roboclaw.DutyM1(self.address, (self.m1_pwm - 64) * int(32767 / 64))
                    self.m1_pwm = None
                if self.m2_pwm is not None:
                    roboclaw.ForwardBackwardM2(self.address, self.m2_pwm)
                    roboclaw.DutyM2(self.address, (self.m2_pwm - 64) * int(32768 / 64))
                    self.m2_pwm = None

                if self.m1_vel is not None:
                    if self.m1_vel == 0:
                        roboclaw.DutyM1(self.address, 0)
                    else:
                        roboclaw.SpeedM1(self.address, self.m1_vel)
                    self.m1_vel = None
                if self.m2_vel is not None:
                    if self.m2_vel == 0:
                        roboclaw.DutyM2(self.address, 0)
                    else:
                        roboclaw.SpeedM2(self.address, self.m2_vel)
                    self.m2_vel = None

                # TODO: Move QPPS to configuration file per motor
                if self.m1_pos is not None:
                    roboclaw.SpeedAccelDeccelPositionM1(self.address, 0, self.QPPS, 0, self.m1_pos, False)
                    self.m1_pos = None
                if self.m2_pos is not None:
                    roboclaw.SpeedAccelDeccelPositionM2(self.address, 0, self.QPPS, 0, self.m2_pos, False)
                    self.m2_pos = None
        except Exception as e:
            rospy.logerr("SIGLARAM: " + str((self.name, os.getpid())) + ": " + str(e))


    def connect(self):
        ports = serial.tools.list_ports.comports()
        random.shuffle(ports)
        for usb in ports:
            try:
                rospy.logerr('about to open ' + str(usb.device))
                if usb.product != 'USB Roboclaw 2x60A':
                    continue
                # Open up the serial port and see if data is available at the desired address
                roboclaw.Open(usb.device, 38400)
                c1, c2 = roboclaw.GetConfig(self.address)
                print(c1)
                print(c2)
                print(' ')
                if (c1 is not 0) or (c2 is not 0):
                    self.device = usb.device
                    rospy.logerr('%s (%s) has connected to roboclaw with address %s', self.device, self.name, self.address)
                    #print(self.name + ' ' + self.device)
                    return True
                else:
                    roboclaw.port.close()
            except IOError:
                rospy.logerr('IOError')
                continue
            # time.sleep(0.1)
        return False


    def callbackM1_pwm(self, msg):
        if self.connected:
            self.timeout = int(round(time.time() * 1000))
            self.m1_pwm = msg.data
        else:
            rospy.loginfo('%s recieved M1_pwm, not connected', self.address)
            pass
            
    def callbackM2_pwm(self, msg):
        if self.connected:
            self.timeout = int(round(time.time() * 1000))
            self.m2_pwm = msg.data
        else:
            rospy.loginfo('%s recieved M2_pwm, not connected', self.address)
            pass


    def callbackM1_vel(self, msg):
        if self.connected:
            self.timeout = int(round(time.time() * 1000))
            self.m1_vel = msg.data
        else:
            # rospy.logerr('%s recieved M1_vel, not connected', self.address)
            pass

    def callbackM2_vel(self, msg):
        if self.connected:
            self.timeout = int(round(time.time() * 1000))
            self.m2_vel = msg.data
        else:
            # rospy.logerr('%s recieved M2_vel, not connected', self.address)
            pass

    def callbackM1_pos(self, msg):
        if self.connected:
            self.timeout = int(round(time.time() * 1000))
            self.m1_pos = msg.data
        else: 
            pass

    def callbackM2_pos(self, msg):
        if self.connected:
            self.timeout = int(round(time.time() * 1000))
            self.m2_pos = msg.data
        else: 
            pass


if __name__ == '__main__':
    # sigalarm is used as a watchdog timer, if the main loop takes more than a second to run then it is interupted
    signal.signal(signal.SIGALRM, SIGALARM_handler)

    rospy.init_node('motornode', anonymous=True)
    name = rospy.get_param('~controller_name', 'asdf')
    m1Name = rospy.get_param('~m1_name', 'M1')
    m2Name = rospy.get_param('~m2_name', 'M2')
    pub_enc = rospy.get_param('~pub_enc', False)
    address = int(rospy.get_param('~address', 0x80))
    controller = motornode(name, m1Name, m2Name, pub_enc, address)
    
    rospy.spin()
