#!/usr/bin/python

#File Name main.py
#Authors Matthew Raneri

'''
Publishes input from the logitech controller (should be device js2)
'''
'''
How do you call this node?
rosrun logitech_controller main.py
'''

# Topics this node is publishing to:
# logitech controller
# publishes a string with a space, first parameter is button name or axis name
# second parameter is the value it outputs

import rospy
import joystick_read

from geometry_msgs.msg import Vector3

joyname = 'js0'

# publish controller info

class joynode:
	pub = None
	jsdev = None
	x=0
	y=0
	z=0
	k=0

	def __init__(self):
		self.pub = rospy.Publisher("beat", Vector3, queue_size=10)
    		rospy.init_node("logitech_controller", anonymous=True)
   		self.jsdev = joystick_read.connect(joyname)
    		self.x=0
    		self.y=0
    		self.z=0
    		self.k=25
		
	def loop(self):
		rate = rospy.Rate(200) # 200hz cycle
    		while not rospy.is_shutdown():
        		# process controller input
        		inputinfo = joystick_read.wait_for_changes(self.jsdev)
			if inputinfo[0]=='y':
				self.x=float(inputinfo[1])*-self.k
			elif inputinfo[0]=='x':
				self.y = float(inputinfo[1])*self.k
			elif inputinfo[0]=='rx':
				self.z = float(inputinfo[1])*-self.k
			self.pub.publish(Vector3(self.x,self.y,self.z))
			print("published x %i, y %i, z %i"%(self.x,self.y,self.z))
        		rate.sleep()

if __name__ == "__main__":
	p = joynode()
	p.loop()
