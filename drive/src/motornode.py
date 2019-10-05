#!/usr/bin/env python

"""
This node communicates with a roboclaw (our red motor controllers). This node lookds really 
complicated because of all the error handling. The roboclaws are really hard to use because sometimes
serial packets are missed and then the code to read data from the roboclaws will hang.
"""

import rospy
import roboclaw
from std_msgs.msg import Int8
from std_msgs.msg import Int16
from std_msgs.msg import Int32
from std_msgs.msg import Float32

import serial.tools.list_ports
import time
import pdb
import signal
import os
import random
import threading, sys, traceback


def SIGALARM_handler(signum, frame):
    """
    Signals are a Unix concept, basically process can recieve messages, called signals, in the form of a single number.
    Different signals do different things. For example, if a process recieves signal 9 (SIGKILL) it is instantly killed.
    This function is called whenever 14 (SIGALRM) is sent. SIGALRM is used to wake a process up after some set
    amount of time.

    If this function is called it normally means that a serial packet was missed or that there's a bug.
    """
    print "watchdog ran out!"
    stacks = dumpstacks(signum, frame)
    rospy.logerr(str(os.getpid()) + ": serial timeout, call stack written to log")
    print stacks

    # The call stack is a little complicated when this line runs. This function is called by the main thread, which 
    # is probably waiting to recieve a missed serial packet (which will never come). When this function throws an exception
    # it rises up the call stack and interrupts the main thread (which is waiting for a serial packet).
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

    # maximum motor speed while in position control mode (expressed in encoder pulses per second).
    # TODO: Move QPPS to configuration file per motor
    QPPS = 500

    name = ''
    address = 0x80
    device = ''

    timeout = 0

    # how many "encoder ticks" the motors have rotated so far
    m1_enc_pub = None
    m2_enc_pub = None

    connected = False

    # the PWM to be sent to the motor (PWM control won't be used if None)
    m1_pwm = None
    m2_pwm = None

    # target motor velocity (velocity control won't be used if None)
    m1_vel = None
    m2_vel = None

    # target motor position (expressed in encoder ticks) (position won't be used if None)
    m1_pos = None
    m2_pos = None


    def __init__(self, name, m1_name='M1', m2_name='M2', publish_enc=False, publish_amps=True, address=0x80):
        self.name = name
        self.address = address
        self.timeout = int(round(time.time() * 1000))

        #PWM from 0 to 127
        rospy.Subscriber(m1_name + '_pwm', Int8, self.callbackM1_pwm)
        rospy.Subscriber(m2_name + '_pwm', Int8, self.callbackM2_pwm)

        #Signed velocity in encoder ticks/second
        rospy.Subscriber(m1_name + '_vel', Int16, self.callbackM1_vel)
        rospy.Subscriber(m2_name + '_vel', Int16, self.callbackM2_vel)

        # signed position in encoder ticks
        rospy.Subscriber(m1_name + '_pos', Int32, self.callbackM1_pos)
        rospy.Subscriber(m2_name + '_pos', Int32, self.callbackM2_pos)

        # set up publishers for the current position of the motor
        if publish_enc:
            self.m1_enc_pub = rospy.Publisher(m1_name + '_enc', Int32, queue_size = 1)
            self.m2_enc_pub = rospy.Publisher(m2_name + ("_enc2" if m1_name == m2_name else "_enc"), Int32, queue_size = 1)
        
        # set up publishers for motor current draws and battery voltage
        if publish_amps:
            self.m1_amp_pub = rospy.Publisher(m1_name + '_amps', Float32, queue_size = 1)
            self.m2_amp_pub = rospy.Publisher(m2_name + ("_amps2" if m1_name == m2_name else "_amps"), Float32, queue_size = 1)
            self.battery_voltage_pub = rospy.Publisher(m2_name + '_battery_voltage', Float32, queue_size = 1)

        # Try to connect to the roboclaw. This is a very error-prone process, especially when there are a lot of roboclaws connected.
        # see the self.connect() method for details
        while not rospy.is_shutdown():
            self.connected = self.connect()
            if self.connected:
                rospy.loginfo('%s has connected (' + str((self.name, os.getpid())) + ')', self.address)
                break
            rospy.sleep(1.0)

        # Loop sending commands to the roboclaw. This is done on a loop instead of when new ROS messages are recieved because
        # the roboclaws have timeouts built in, so they continually have to be sent commands.
        update_rate = 0.05
        while not rospy.is_shutdown():
            start_time = time.time()
            self.update()
            time.sleep(max(start_time + update_rate - time.time(), 0))

    def update(self):
        """
        Sends
        """
        signal.setitimer(signal.ITIMER_REAL, 0.25)  # set the watchdog for 0.25 seconds (SIGALARM_handler() will be
                                                    # called if this function takes too long)
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

            # Publish current readings
            if self.m1_enc_pub is not None:
                currents = roboclaw.ReadCurrents(self.address)
                if currents[0] == 0:
                    rospy.logerr(str((self.name, os.getpid())) + ": error returned from current reading: " + str(currents))
                else:
                    m1_amp_msg = Float32()
                    m1_amp_msg.data = currents[1] / 100.0
                    m2_amp_msg = Float32()
                    m2_amp_msg.data = currents[2] / 100.0
                    self.m1_amp_pub.publish(m1_amp_msg)
                    self.m2_amp_pub.publish(m2_amp_msg)

                bat_voltage = roboclaw.ReadMainBatteryVoltage(self.address)
                if bat_voltage[0] == 0:
                    rospy.logerr(str((self.name, os.getpid())) + ": error returned from battery voltage reading: " + str(bat_voltage))
                else:
                    bat_volt_msg = Float32()
                    bat_volt_msg.data = bat_voltage[1] / 10.0
                    self.battery_voltage_pub.publish(bat_volt_msg)

            # Timeout if no command recieved for more than TIMEOUT_TIME
            if int(round(time.time() * 1000)) - self.timeout > self.TIMEOUT_TIME:
                roboclaw.ForwardBackwardM1(self.address, 64)
                roboclaw.ForwardBackwardM2(self.address, 64)
            else:
                # PWM control
                if self.m1_pwm is not None:
                    roboclaw.ForwardBackwardM1(self.address, self.m1_pwm)
                    # roboclaw.DutyM1(self.address, (self.m1_pwm - 64) * int(32767 / 64))
                    self.m1_pwm = None
                if self.m2_pwm is not None:
                    roboclaw.ForwardBackwardM2(self.address, self.m2_pwm)
                    # roboclaw.DutyM2(self.address, (self.m2_pwm - 64) * int(32768 / 64))
                    self.m2_pwm = None

                # Velocity control
                if self.m1_vel is not None:
                    # if the commanded velocity is 0 then turn the motors off instead of sending a zero speed.
                    # This is to keep the motors from overheating, we once left the rover still on a slight slope for
                    # a few minutes and the motors overheated trying to keep the rover still.
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

                # Position control
                if self.m1_pos is not None:
                    roboclaw.SpeedAccelDeccelPositionM1(self.address, 0, self.QPPS, 0, self.m1_pos, False)
                    self.m1_pos = None
                if self.m2_pos is not None:
                    roboclaw.SpeedAccelDeccelPositionM2(self.address, 0, self.QPPS, 0, self.m2_pos, False)
                    self.m2_pos = None
        
        # this exception will be raised by the SIGALRM handler if sending/recieving data from the roboclaw took too long
        except Exception as e:
            rospy.logerr("SIGLARAM: " + str((self.name, os.getpid())) + ": " + str(e))


    def connect(self):
        """
        If there are multiple roboclaws connected to the computer then there is no good way to tell which one is which
        without connecting to them all. That's basically what this function does. It gets a list of all attached
        Roboclaws then connects to them one-by-one and reads their device ID.

        We distinguish between Roboclaws through their user-settable address. 

        If another instance of motornode.py is currently connected to a roboclaw then no other instances can connect
        to that specific roboclaw until the other instance closes the connect. That's why this function needs to run
        in a loop, another instance might be connected to this instance's roboclaw so it'll get skipped. 
        """
        ports = serial.tools.list_ports.comports()
        random.shuffle(ports)
        for usb in ports:
            try:
                if usb.product != 'USB Roboclaw 2x60A':
                    continue
                rospy.logerr('about to open ' + str(usb.device))
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
    pub_amps = rospy.get_param('~pub_amps', False)
    address = int(rospy.get_param('~address', 0x80))
    controller = motornode(name, m1Name, m2Name, pub_enc, pub_amps, address)
    
    rospy.spin()
