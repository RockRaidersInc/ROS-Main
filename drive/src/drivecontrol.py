#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3
from std_msgs.msg import String


class drivecontrol:
	#constant for wheel stopped value
	stop = 64.0
	#constants for wheel direction/steering
	k1 = 1.0
	k2 = 1.0
	k3 = -1.0
	k4 = 1.0
	k5 = 2.0
	k6 = 2.0

	pubf = None
	pubs = None
	pubb = None

	def callback(self,data):
    		x = int(data.x)
    		y = int(data.y)
    		z = int(data.z)
    		rospy.loginfo("recieved: %i,%i,%i"%(x,y,z))
   
		#wheel logic
    		fr = self.stop + self.k1*data.x
    		fl = self.stop + self.k2*data.x
    		br = self.stop + self.k3*data.x
    		bl = self.stop + self.k4*data.x

		#steering logic
    		l = self.k5*y+self.k6*z + self.stop
    		r = self.k6*z-self.k5*y + self.stop

		#publish stuff
    		Front = Vector3(fr,fl,0)
    		Back = Vector3(br,bl,0)
    		Steer = Vector3(l,r,0)
    
    		self.pubf.publish(Front)
    		self.pubs.publish(Steer)
    		self.pubb.publish(Back)

	def changevar(self,data):
		command = data.data
		com = command.split(' ')
		if com[0]=='k1':
			self.k1=float(com[1])
		elif com[0]=='k2':
			self.k2=float(com[1])
		elif com[0]=='k3':
			self.k3=float(com[1])
		elif com[0]=='k4':
			self.k4=float(com[1])
		elif com[0]=='k5':
			self.k5=float(com[1])
		elif com[0]=='k6':
			self.k6=float(com[1])
		elif com[0]=='print':
			print("k1=%f,k2=%f,k3=%f,k4=%f,k5=%f,k6=%f"%(self.k1,self.k2,self.k3,self.k4,self.k5,self.k6))


	def __init__(self):
    		rospy.init_node('DriveController', anonymous=True)

    		rospy.Subscriber('drive', Vector3, self.callback)
		rospy.Subscriber('drivecontrol',String, self.changevar)
    
    		self.pubf = rospy.Publisher('front', Vector3, queue_size=1)
		self.pubs = rospy.Publisher('steer', Vector3, queue_size=1)
		self.pubb = rospy.Publisher('back', Vector3, queue_size=1)

    		# spin() simply keeps python from exiting until this node is stopped
    		rospy.spin()

if __name__ == '__main__':
    d = drivecontrol()
