#!/usr/bin/env python

##############################################################################

#File name: tennis_detection.py
#Authors: Nick

'''
Uses opencv to detect tennis balls
'''
'''
How do you call this node?
rosrun <vision> <tennis_detection> <parameters>
'''

#Topics this node is subscribed to: webcam?
#Topics this node publishes to
#Services this node uses
#Other dependencies?

##############################################################################

#include
import rospy
import cv2
import numpy as np
from time import sleep as wait

#CONSTANTS (organize these as necessary)
#names for constants should be in ALL CAPS

##############################################################################

#Setup
#every node should have one
def Setup():

    cap = cv2.VideoCapture('/home/nick/catkin_ws/src/ROS-Main/vision/src/TennisBalls.mp4')
    yellow = np.uint8([[[255,255,0]]])
    hsv_yellow = cv2.cvtColor(yellow,cv2.COLOR_BGR2HSV)
    print hsv_yellow
    return cap

#Loop
#every node should have one
def Loop(cap):
    while(True):
        # Take each frame
        _,frame = cap.read()
        if frame == None:
            return
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of yellow color in HSV
        lower_yellow = np.array([0,100,0])
        upper_yellow = np.array([50,255,255])

        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        '''
        gray = cv2.medianBlur(frame,5)
        gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAhY)
        '''

        mask = cv2.medianBlur(mask,5)


        circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,25,param1=100,param2=12,minRadius=0,maxRadius=50)
        if circles!=None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                
                cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
                
        cv2.imshow('Detected tennis balls',frame)
        cv2.imshow('Mask',mask)
        cv2.imshow('Res',res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            return

        #Wait for a moment to destroy frame
        wait(0.05)

##############################################################################

#Helper Functions

'''
function header
what does it return? what parameters? general description.
'''

#TODO: Add some way to calibrate for the color white to be able to compensate for other lighting conditions

def Foo():
    pass
    '''
    body of function
    MAKE SURE YOUR EDITOR USES 4 SPACES FOR TABS
    '''

##############################################################################

if __name__ == '__main__':
    Loop(Setup())
