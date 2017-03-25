#!/usr/bin/python

import rospy
import roscpp
import rostopic
import rosgraph
from Tkinter import *
import serial.tools.list_ports
from sensor_msgs.msg import Image



class topicdisp:

	topics = []
	frame = None

	def __init__(self,root,x,y):
		self.frame = Frame(root)
		self.frame.grid(row = y, column = x)

		self.printButton = Button(self.frame, text = "update Topics", command = self.Topics)
		self.printButton.grid(row = 1, column = 0)

		self.label = Label(self.frame, text="Topics")
		self.label.grid(row=0, column=0)
		


	def Topics(self):
		self.topics = [t for t, _ in rosgraph.Master('/mynode').getPublishedTopics('')]
		x = 1
		for t in self.topics:
			entry = Label(self.frame, text = str(t))
			entry.grid(row=0, column = x)
			x+=1


class camerafeed:
	frame = None

	def __init__ (self, root, x,y, topic):
		self.frame = Frame(root)
		self.frame.grid(row = y, column = x)

		self.image = PhotoImage(self.frame)
		


class usblist:
	ports = []

	frame = None
	printButton = None
	label = None
	
	def __init__(self,root,x,y):
		self.frame = Frame(root)
		self.frame.grid(row = y, column = x)

		self.printButton = Button(self.frame, text = "update USBs", command = self.refresh)
		self.label = Label(self.frame, text="USBs:")
		self.refresh()

	def standard(self):
		self.printButton.grid(row = 1, column = 0)
		self.label.grid(row=0, column=0)
	
	def refresh(self):
		self.ports = serial.tools.list_ports.comports()
		x=1
		#self.frame.grid_forget()
		self.standard()
		for dev in self.ports:
			entry = Label(self.frame, text = dev.name)
			entry.grid(row = 0,column= x)
			x+=1
		if (len(self.ports) < 1):
			entry = Label(self.frame, text = "none attached")
			entry.grid(row=0,column = 1)
		

