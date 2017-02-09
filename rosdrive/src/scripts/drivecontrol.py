#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

import rospy
from geometry_msgs.msg import Vector3

stop = 64.0

k1 = 1.0
k2 = -1.0
k3 = 1.0
k4 = -1.0
k5 = 1.0
k6 = -1.0
kt = 1.0
pubf = rospy.Publisher('Front', Vector3, queue_size=1)
pubm = rospy.Publisher('Mid', Vector3, queue_size=1)
pubb = rospy.Publisher('Back', Vector3, queue_size=1)

def callback(data):
    x = int(data.x)
    y = int(data.y)
    rospy.loginfo("recieved: forward %i, clockwise: %i"%(x,y))
    
    fr = stop + k1*data.x +kt*data.y
    fl = stop + k2*data.x +kt*data.y
    mr = stop + k3*data.x +kt*data.y
    ml = stop + k4*data.x +kt*data.y
    br = stop + k5*data.x +kt*data.y
    bl = stop + k6*data.x +kt*data.y
    
    Front = Vector3(fr,fl,0)
    Mid = Vector3(mr,ml,0)
    Back = Vector3(br,bl,0)
    
    pubf.publish(Front)
    pubm.publish(Mid)
    pubb.publish(Back)


def driveControl():
    rospy.init_node('DriveController', anonymous=True)

    rospy.Subscriber('Drive', Vector3, callback)
    
    

    

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    driveControl()
