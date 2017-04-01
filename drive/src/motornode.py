#!/usr/bin/env python

import rospy
import roboclaw
from geometry_msgs.msg import Vector3
from std_msgs.msg import String


class motornode:
	name = ''
	address = 0x80
	device = "/dev/ttyACM0"
	x=64
	y=64
	timeout = 1000

	pub = None

	def __init__(self,n):
		self.name = n
		rospy.init_node('motornode', anonymous=True)

    		self.pub = rospy.Publisher('usb', String, queue_size = 1)

    		rospy.Subscriber(self.name, Vector3, self.callback)
    		rospy.Subscriber(self.name+'device', String, self.setdevice)
    
    		print("start finished")

    		rospy.spin()
    	def callback(self,data):
    		self.timeout = 1000

    		self.x = int(data.x)
    		self.y = int(data.y)
    		print("(x,y) = (%i,%i)"%(self.x,self.y))



	def setdevice(self,data):
    		self.device = str(data.data)
    		rospy.loginfo("device set to: "+self.device)
    		self.connect()

	def connect(self):
		print("trying to connect")
		connect = False
        	wait = rospy.Rate(1)
		while (connect==False):
    			try:
				print self.device
    				roboclaw.Open(self.device,115200)
       		 		connect = True
				self.pub.publish("connect "+self.device)
    			except:
				print("no device "+ self.device)
				self.pub.publish("disconnect "+ self.device)
			wait.sleep()
		self.normal()

	def setmotor (self,m,n):
   	 try:
      	 	roboclaw.ForwardBackwardM1(self.address,m)
      	 	roboclaw.ForwardBackwardM2(self.address,n)

   	 except:
		print("cannot update motors, possible disconnect")
		self.pub.publish ("disconnect "+ self.device)
		self.connect()
	
	def normal(self):
  		while not rospy.is_shutdown():
			if (self.timeout > 0):
				print("setting motor to values %i, %i"%(self.x,self.y))
      				self.setmotor(self.x,self.y)
       				self.timeout -=1
        		else:
	    			self.setmotor(64,64)

    
	

if __name__ == '__main__':
    controller = motornode('front')
