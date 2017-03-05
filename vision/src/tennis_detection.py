import cv2
import numpy as np

cap = cv2.VideoCapture('TennisBalls.mp4')
yellow = np.uint8([[[255,255,0 ]]])
hsv_yellow = cv2.cvtColor(yellow,cv2.COLOR_BGR2HSV)
print hsv_yellow

while(1):

    # Take each frame
    _, frame = cap.read()
    if frame == None:
        break

    gray = cv2.medianBlur(frame,5)
    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,40,param1=200,param2=40,minRadius=0,maxRadius=0)
    if circles!=None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
        	cv2.circle(gray,(i[0],i[1]),i[2],(0,255,0),2)
        	cv2.circle(gray,(i[0],i[1]),2,(0,0,255),3)
	
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of yellow color in HSV
    lower_yellow = np.array([0,100,0])
    upper_yellow = np.array([50,255,255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.imshow('detected circles',gray)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
