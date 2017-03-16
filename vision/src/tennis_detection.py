import cv2
import numpy as np
from time import sleep as wait

cap = cv2.VideoCapture('TennisBalls.mp4')
yellow = np.uint8([[[255,255,0 ]]])
hsv_yellow = cv2.cvtColor(yellow,cv2.COLOR_BGR2HSV)
print hsv_yellow

while(1):

    # Take each frame
    _, frame = cap.read()
    if frame == None:
        break

	
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
    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    '''
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,25,param1=100,param2=12,minRadius=0,maxRadius=50)
    if circles!=None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
            

    cv2.imshow('Detected tennis balls',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('Res',res)
    #cv2.imshow('detected circles',gray)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    #Wait for a moment to destroy frame
    wait(0.05)



cv2.destroyAllWindows()
