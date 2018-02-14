#!/usr/bin/env python
import rospy
import serial.tools.list_ports
from std_msgs.msg import String
import sys
class device:
	name = ""
	pub = None
	loc = ""
	device = ""

	def __init__(self,n, p, l):
		self.name = n
		self.pub = p
		self.loc = l
	

class devicenode:

	con = {}
	req = {}
	jobs = []
	wait = None
	
	def __init__(self):
		rospy.init_node('deviceconnect', anonymous=True)
		print("test")
		rospy.Subscriber('usb', String, self.call)
	
		f = open("files.txt","r")
	
		x=1
		for line in f:
			if line[0]!='#':
				line = line[0:-1]
				dev = line.split(',')
				if len(dev) !=2:
					print("error with line %i"%x)
					print(line)
				else:
					pub = rospy.Publisher(dev[0],String, queue_size = 1)
					self.jobs.append(device(dev[0],pub,dev[1]))
					print("Searching for: "+ dev[0])
			x+=1

	

		
	
		self.wait = rospy.Rate(1)
		x = 0
		while x < 5:
			self.wait.sleep()
			x+=1	
		self.loop()




	def call(self,data):

		command = data.data
		com = command.split(' ')
		print (com[0])
		print (com[1])
		if com[0] == 'connect':
			try:
				print("connecting")
				dev = self.req[com[1]]
				print(dev.device)
				self.req.pop(dev.device)
				self.con[com[1]]= dev
			except:
				print "no requested device by name: "+com[1]
			


		elif com[0] == 'disconnect':

			try:
				dev = self.con[com[1]]
				self.con.pop(dev.device)
				self.jobs.append(dev)
			except:
				print("fuck you and everyone around you no connected device by name: "+com[1])
		
		else:
			print ('recieved invalid command type')
			print command	


	def loop(self):

		while not rospy.is_shutdown():
			#sys.stderr.write("\x1b[2J\x1b[H")
			ports = serial.tools.list_ports.comports()
			
			av = {}
			for usb in ports:
				av[usb.location]= usb.device
				#print (usb.location + " " +req[0].loc)
				#print ("This is a test "+str(req[0].loc == usb.location))
			print("looking for devices:")
			x=0
			for dev in self.jobs:
				print "\t"+dev.name
				x+=1


			print "\nrequested devices:"
			
			for dev in self.req.itervalues():
				print "\t"+ dev.name + " "+ dev.device
				x+=1
			
			print "\nmatched devices:"
			x=0
			for dev in self.con.itervalues():
				print "\t"+ dev.name + " "+ dev.device
				x+=1
			x=0
			
			print "\nconnected usbs:"
			print (av)
			#print self.jobs
			#print self.req
			#print self.con
			print "\n"
		#	print x
			q = []

	
			name = ""
			for dev in self.jobs:
				try:
					#print ("entered try")
					name = av[dev.loc]
					#print ("name == "+name)
					
					#print("name was published")
					
					dev.device = name
					self.req[dev.device] = dev
					self.jobs.remove(dev)
					
					#print ("made it through loop")

				except:
					x+=1
					print("no usb connected at location "+ dev.loc)

			for dev in self.req.itervalues():
				dev.pub.publish(String(dev.device))
				print("request: %s to %s"%(dev.name, dev.device))
			"""q = list(self.req.itervalues())
			for z in q:
				try:
					
					asdf = av[z.loc]
				except:
					self.jobs.append(self.req[z.device])
					self.req.pop(z.device)
			q = list(self.con.itervalues())"""


			self.wait.sleep()


if __name__ == "__main__":
	d =devicenode()
