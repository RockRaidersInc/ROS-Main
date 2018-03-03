#!/usr/bin/env python

import rospy
import roboclaw
from geometry_msgs.msg import Vector3
from std_msgs.msg import String
from std_msgs.msg import Int8


class motornode:
	name = ''
	address = 0x80
	device = "/dev/00000"
	x=64
	y=64
	timeout = 1000
	pub = None
	enc_pub = None

	def __init__(self,n,m1_name='M1',m2_name='M2',useEnc=False):
		self.name = n

		self.pub = rospy.Publisher('usb', String, queue_size = 1)

		rospy.Subscriber(self.name, Vector3, self.callback)
		rospy.Subscriber(self.name+'device', String, self.setdevice)
		rospy.Subscriber(m1_name, Int8, self.callbackM1)
		rospy.Subscriber(m2_name, Int8, self.callbackM2)
		
		if (useEnc == True):
			self.enc_pub = rospy.Publisher(self.name+'encoder', Vector3, queue_size=3)
    
		while not rospy.is_shutdown():
			self.connect()
			self.normal(useEnc)
		self.connect()

	def callback(self,data):
		self.timeout = 1000

		self.x = int(data.x)
		self.y = int(data.y)

	def callbackM1(self, msg):
		self.timeout = 1000
		self.x = msg.data

	def callbackM2(self, msg):
		self.timeout = 1000
		self.y = msg.data



	def setdevice(self,data):
		self.device = str(data.data)
		rospy.loginfo("device set to: "+self.device)
			

	def connect(self):
		print("trying to connect")
		connect = False
		wait = rospy.Rate(1)
		while (connect==False):
				try:
					print(self.device)
					roboclaw.Open(self.device,115200)
					connect = True
					self.pub.publish("connect "+self.device)
					roboclaw.ResetEncoders(self.address);
					print ("connected")
					break
				except:
					print("no device "+ self.device)
					self.pub.publish("disconnect "+ self.device)
					wait.sleep()

	def setmotor (self,m,n):
                rospy.logdebug("Setting motor to %d, %d", m, n)
		roboclaw.ForwardBackwardM1(self.address,m)
		roboclaw.ForwardBackwardM2(self.address,n)

	 
		
	
	def normal(self, useEnc):
		while not rospy.is_shutdown():
			if (self.timeout > 0):
				#print("setting motor to values %i, %i"%(self.x,self.y))
				self.enc(useEnc)
				try:
					self.setmotor(self.x,self.y)
				except:
					er = 1
					self.timeout -=1
					break
			else:
				try:
					self.setmotor(64,64)
				except:
					er = 1
					break
    
	
	def enc(self, useEnc):
		if (useEnc == False):
			return
		enc1 = roboclaw.ReadEncM1(self.address)
		enc2 = roboclaw.ReadEncM2(self.address)

		self.enc_pub.publish(Vector3(enc1[1], enc2[1], 0))

if __name__ == '__main__':
	rospy.init_node('motornode', anonymous=True)
	name = rospy.get_param('~controller_name', 'motornode')
	m1Name = rospy.get_param('~m1_name', 'M1')
	m2Name = rospy.get_param('~m2_name', 'M2')
        use_enc = rospy.get_param('~use_enc', False)
	controller = motornode(name, m1Name, m2Name, use_enc)
