# Basically Shu-Nong's Trackbar script with Hough Circle Transforms - xieo
# Code not yet integrated with ROS

import numpy as np
import cv2
import sys

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

def main():
    print('Creating settings window')
    createSettingsWindow()

    img_name = str(sys.argv[1])
    print('Image name: {}'.format(img_name))

    while True:
        # Get the frame
        frame = cv2.imread(img_name, cv2.IMREAD_COLOR)

        # HSV Color Filter Params
        hl = cv2.getTrackbarPos("H_low","Settings")
        hh = cv2.getTrackbarPos('H_high','Settings')
        sl = cv2.getTrackbarPos('S_low','Settings')
        sh = cv2.getTrackbarPos('S_high','Settings')
        vl = cv2.getTrackbarPos('V_low','Settings')
        vh = cv2.getTrackbarPos('V_high','Settings')
        
        # Color Filter
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([hl,sl,vl])
        upper_hsv = np.array([hh,sh,vh])
        result = cv2.inRange(hsv_frame,lower_hsv,upper_hsv)

        # Show Images
        cv2.imshow('Original',frame)
        cv2.imshow('Result',result)

        # Loop every 30ms, if press key end program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



