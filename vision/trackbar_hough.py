# Basically Shu-Nong's Trackbar script with Hough Circle Transforms - xieo
# Code not yet integrated with ROS

import numpy as np
import cv2
import sys

def usageStatement():
    print("Usage: python2 %s [video_file]", sys.argv[0])

# Empty callback
def nothing(x):
    pass

def createSettingsWindow():
    cv2.namedWindow("Settings",flags=cv2.WINDOW_NORMAL)
    # Add trackbars here
    cv2.createTrackbar('H_low','Settings',20,255,nothing)
    cv2.createTrackbar('H_high','Settings',64,255,nothing)
    cv2.createTrackbar('S_low','Settings',75,255,nothing)
    cv2.createTrackbar('S_high','Settings',255,255,nothing)
    cv2.createTrackbar('V_low','Settings',75,255,nothing)
    cv2.createTrackbar('V_high','Settings',255,255,nothing)

    cv2.createTrackbar('ksize','Settings',5,100,nothing)

    cv2.createTrackbar('#erode','Settings',2,10,nothing)
    cv2.createTrackbar('#dilate','Settings',0,10,nothing)
    
    cv2.createTrackbar("inverse_accumulator_res","Settings",1,5,nothing)
    cv2.createTrackbar("min_centers_distance_percent","Settings",100,100,nothing)
    cv2.createTrackbar("canny_edge_high_thresh","Settings",150,255,nothing)
    cv2.createTrackbar("accumulator_center_thresh","Settings",7,255,nothing)
    cv2.createTrackbar("min_circle_radius_percent","Settings",0,100,nothing)
    cv2.createTrackbar("max_circle_radius_percent","Settings",0,100,nothing)
    
# Get the camera/video file
if(len(sys.argv) > 1):
    print("opening video file")
    cap = cv2.VideoCapture(sys.argv[1])
else:
    print("opening webcam?")
    cap = cv2.VideoCapture(0)    
if not cap.isOpened():
    usageStatement()
    sys.exit("Could not open camera/video")

createSettingsWindow()
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 600,600)
cv2.namedWindow("processed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("processed", 600,600)
cv2.namedWindow("greyscale", cv2.WINDOW_NORMAL)
cv2.resizeWindow("greyscale", 600,600)
cv2.namedWindow("blur", cv2.WINDOW_NORMAL)
cv2.resizeWindow("blur", 600,600)
cv2.namedWindow("filtered_mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("filtered_mask", 600,600)
cv2.namedWindow("erode", cv2.WINDOW_NORMAL)
cv2.resizeWindow("erode", 600,600)
cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
cv2.resizeWindow("dilate", 600,600)

cv2.waitKey()
red = (0,0,255)
green = (0,255,0)

while True:
    # Get the frame & return value
    ret, frame = cap.read()
    # Validate
    if not ret:
        print("Error: No Frame Read!")
        break

    # Get parameters from the trackbars

    # HSV Color Filter Params
    hl = cv2.getTrackbarPos("H_low","Settings")
    hh = cv2.getTrackbarPos('H_high','Settings')
    sl = cv2.getTrackbarPos('S_low','Settings')
    sh = cv2.getTrackbarPos('S_high','Settings')
    vl = cv2.getTrackbarPos('V_low','Settings')
    vh = cv2.getTrackbarPos('V_high','Settings')

    # Erode and Dilate Params
    er = cv2.getTrackbarPos('#erode', 'Settings')
    di = cv2.getTrackbarPos('#dilate', 'Settings')

    # Median Blur
    ksize = cv2.getTrackbarPos("ksize", "Settings")*2 + 1
    
    # Hough Circles Params
    rat = cv2.getTrackbarPos("inverse_accumulator_res","Settings")
    sep = cv2.getTrackbarPos("min_centers_distance_percent","Settings")
    thh = cv2.getTrackbarPos("canny_edge_high_thresh","Settings")
    tha = cv2.getTrackbarPos("accumulator_center_thresh","Settings")
    rdl = cv2.getTrackbarPos("min_circle_radius_percent","Settings")
    rdh = cv2.getTrackbarPos("max_circle_radius_percent","Settings")
    
    # Apply operations on images

    # Color Filter
    lower_hsv = np.array([hl,sl,vl])
    upper_hsv = np.array([hh,sh,vh])
    # Convert to HSV colorspace
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Apply Median Blur
    # Median Blur is pretty slow on anything larger than 1080p
    blur_frame_hsv = cv2.medianBlur(hsv_frame,ksize)
    blur_frame_bgr = cv2.medianBlur(frame,ksize)
    # Use if speed matters
    #blur_frame_hsv = cv2.blur(hsv_frame,(ksize, ksize))
    #blur_frame_bgr = cv2.blur(frame,(ksize, ksize))
    
    # Filter 
    filter_mask = cv2.inRange(blur_frame_hsv,lower_hsv,upper_hsv)
    # Transformations
    erode_mask = cv2.erode(filter_mask, None, iterations=er)
    dilate_mask = cv2.erode(erode_mask, None, iterations=di)
    # Apply Mask
    processed = cv2.bitwise_and(blur_frame_bgr,blur_frame_bgr,mask=dilate_mask)
    grey = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    height, width, channels = processed.shape 

    # print(rat, int(sep/100.0 * width), thh, tha, int(rdl/100.0 * width), int(rdh/100.0 * width))
    
    circles = cv2.HoughCircles(grey,
                               cv2.HOUGH_GRADIENT,
                               rat,
                               sep/100.0 * width,
                               param1=thh,
                               param2=tha,
                               minRadius=int(rdl/100.0 * width),
                               maxRadius=int(rdh/100.0 * width))
    result = frame.copy()
    # Check if we have possible circles
    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Prints all detected circles, x,y, and radius
        print(circles)


        for circle in circles[0]:
            # Draw Circles (Each circle)
            cv2.circle(result, (circle[0],circle[1]), circle[2], green, 2)
            # Centers
            cv2.circle(result, (circle[0],circle[1]), 3, red, -1)

    # Show Images
    cv2.imshow("Result",result)
    cv2.imshow("processed", processed)
    cv2.imshow("greyscale", grey)
    cv2.imshow("blur", blur_frame_hsv)
    cv2.imshow("filtered_mask", filter_mask)
    cv2.imshow("erode", erode_mask)
    cv2.imshow("dilate", dilate_mask)

    # Loop every 30ms, if press key end program
    if cv2.waitKey(30) >= 0:
        break

cap.release()
cv2.destroyAllWindows()
