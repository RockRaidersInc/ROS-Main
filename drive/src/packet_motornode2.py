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
    connected = False

    def callbackPwm(self, msg, args):
        if not self.connected:
            return
        i = args
        if self.parameters[i] is not None:
            self.pwms[i] = msg.data
            self.timeouts[i] = int(round(time.time() * 1000))

    def callbackVel(self, msg, args):
        if not self.connected:
            return
        i = args
        if self.parameters[i] is not None:
            self.velocities[i] = msg.data
            #print('updating_vel ' + str(msg.data))
            self.timeouts[i] = int(round(time.time() * 1000))

    def callbackPos(self, msg, args):
        if not self.connected:
            return
        i = args
        if self.parameters[i] is not None:
            self.positions[i] = msg.data
            self.timeouts[i] = int(round(time.time() * 1000))



    def __init__(self, name, device):
        self.name = name
        self.device = device
        # self.timeout = int(round(time.time() * 1000))

        num = rospy.get_param('motors/num', 8)
        self.parameters = [None]*num
        self.enc_publishers = [None]*num
        self.timeouts = [None]*num

        self.positions = [None]*num
        self.velocities = [None]*num
        self.pwms = [None]*num

        for i in range(num):
            param_name = 'motors/' + str(i)
            if rospy.has_param(param_name):
                param = rospy.get_param(param_name)
                #print(param)
                if not all(k in param for k in ['address', 'motor_num', 'type', 'pub_enc', 'name']):
                    rospy.loginfo('Parameter missing for m' + str(i))
                    continue
                self.parameters[i] = param

                '''
                if param['type'] == 'pwm':
                    rospy.Subscriber(param['name']+'_pwm', Int8, self.callbackPwm, (i))
                if param['type'] == 'vel':
                    rospy.Subscriber(param['name']+'_vel', Int16, self.callbackVel, (i))
                if param['type'] == 'pos' and 'qpps' in param:
                    rospy.Subscriber(param['name']+'_pos', Int32, self.callbackPos, (i))
                '''
                if 'pwm' in param['type']:
                    rospy.Subscriber('/motors/' + param['name']+'_pwm', Int8, self.callbackPwm, (i))
                if 'vel' in param['type']:
                    rospy.Subscriber('/motors/' + param['name']+'_vel', Int16, self.callbackVel, (i))
                if 'pos' in param['type'] and 'qpps' in param:
                    rospy.Subscriber('/motors/' + param['name']+'_pos', Int32, self.callbackPos, (i))


                if param['pub_enc']:
                    self.enc_publishers[i] = rospy.Publisher('/motors/' + param['name']+'_enc', Int32, queue_size=1)


        while not rospy.is_shutdown():
            # Try to connect every second
            self.connected = self.connect()
            if self.connected:
                rospy.loginfo(self.name + ' has connected')
                #print("connected")
                break
            rospy.sleep(1.0)

        update_rate = 0.05
        while not rospy.is_shutdown():
            start_time = time.time()
            self.update()
            #print(start_time - time.time())
            #time.sleep(max(start_time + update_rate - time.time(), 0))

    def update(self):
        for i, pub in enumerate(self.enc_publishers):
            if pub is None:
                continue
            address = self.parameters[i]['address']
            if self.parameters[i]['motor_num'] == 1:
                response = roboclaw.ReadEncM1(address)
            elif self.parameters[i]['motor_num'] == 2:
                response = roboclaw.ReadEncM2(address)
            else:
                continue

            msg = Int32()
            msg.data = int(response[1])
            pub.publish(msg)

        for i, t in enumerate(self.timeouts):
            if self.enc_publishers[i] is None:
                continue
            if int(round(time.time() * 1000)) - t > self.TIMEOUT_TIME:
                address = self.parameters[i]['address']
                if self.parameters[i]['motor_num'] == 1:
                    roboclaw.ForwardBackwardM1(address, 64)
                elif self.parameters[i]['motor_num'] == 2:
                    roboclaw.ForwardBackwardM2(address, 64)
                else:
                    continue

        '''
        for i, pwm in enumerate(self.pwms):
            if pwm is None:
                continue
            if self.parameters[i] is not None:
                address = self.parameters[i]['address']
                if self.parameters[i]['motor_num'] == 1:
                    roboclaw.ForwardBackwardM1(address, pwm)
                elif self.parameters[i]['motor_num'] == 2:
                    roboclaw.ForwardBackwardM2(address, pwm)
                self.pwms[i] = None
        '''


        for i, vel in enumerate(self.velocities):
            if vel is None:
                continue
            #print('found_vel: ' + str(vel))
            if self.parameters[i] is not None:
                address = self.parameters[i]['address']
                #print('address: ' + str(address))
                if self.parameters[i]['motor_num'] == 1:
                    #print('setting m1 speed')
                    if roboclaw.SpeedM1(address, vel):
                        #print('m1_set')
                        pass
                    else:
                        #print('m1_failed')
                        pass
                elif self.parameters[i]['motor_num'] == 2:
                    #print('setting m2 speed')
                    if roboclaw.SpeedM2(address, vel):
                        #print('m1_set')
                        pass
                    else:
                        #print('m1_failed')
                        pass
                self.velocities[i] = None
                

        '''
        for i, pos in enumerate(self.positions):
            if pos is None:
                continue
            if self.parameters[i] is not None:
                address = self.parameters[i]['address']
                qpps = self.parameters[i]['qpps']
                if self.parameters[i]['motor_num'] == 1:
                    roboclaw.SpeedAccelDeccelPositionM1(address, 0, qpps, 0, pos, False)
                elif self.parameters[i]['motor_num'] == 2:
                    roboclaw.SpeedAccelDeccelPositionM2(address, 0, qpps, 0, pos, False)
            self.positions[i] = None
        '''
                
            

    def connect(self):
        try:
            roboclaw.Open(self.device, 38400)
            return True
        except IOError:
            return False


if __name__ == '__main__':
    # sigalarm is used as a watchdog timer, if the main loop takes more than a second to run then it is interupted
    signal.signal(signal.SIGALRM, SIGALARM_handler)

    rospy.init_node('motornode', anonymous=True)
    name = rospy.get_param('~name', 'asdf')
    device = rospy.get_param('~device', '/dev/ttyACM0')
    controller = motornode(name, device)
    
    rospy.spin()
