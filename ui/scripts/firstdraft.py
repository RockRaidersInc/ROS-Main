#!/usr/bin/python

from Tkinter import *
from somepanels import *
		


root = Tk()
topics = topicdisp(root,0,0)
camera = camerafeed(root,0,0,"camera")
usbs = usblist(root,2,0)
root.mainloop()
