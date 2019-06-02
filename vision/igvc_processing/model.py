import numpy as np
import cv2
import sys
import yaml

import rospy
from sensor_msgs.msg import Image
#from cv_bridge import CVBridge, CvBridgeError

#bridge = CvBridge()

class Processing:
    def __init__(self, source_type, source):
        self.source_type = source_type
        self.source = source

        self.frame = None
        if self.source_type == 'image':
            self.frame = cv2.imread(source)
        if self.source_type == 'ros':
            rospy.init_node('trackbar')
            rospy.Subscriber(self.source, Image, self.ros_image_callback)
        if self.source_type == 'video':
            pass
        if self.source_type == 'cap':
            self.cap = cv2.VideoCapture(0)

        #print('hello')
        #print(yaml.load(open(self.SETTING_FILENAME)))
        #print('hello')
        self.settings = {}

    def ros_image_callback(self, msg):
        try:
            self.frame = cv2.resize(bridge.imgmsg_to_cv2(msg, "bgr8"), (640, 480))
        except CvBridgeError as e:
            print("error recieving image\n", e)

    def update_settings(self, settings):
        self.settings = settings

    def hsv_color_filter(self, frame):
        lower_hsv = np.array([self.settings['hl'],self.settings['sl'],self.settings['vl']])
        upper_hsv = np.array([self.settings['hh'],self.settings['sh'],self.settings['vh']])
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result = cv2.inRange(result,lower_hsv,upper_hsv)

        return result

    def average_filter(self, frame):
        s = self.settings['avg']
        result = cv2.blur(frame, (s, s))
        return result

    def gaussian_filter(self, frame):
        s = self.settings['gauss']
        result = cv2.GaussianBlur(frame, (s, s), 0)
        return result

    def median_filter(self, frame):
        s = self.settings['med']
        result = cv2.medianBlur(frame, s)
        return result

    def erode(self, frame):
        s = self.settings['erode']
        result = cv2.erode(frame, None, iterations = s)
        return result

    def dilate(self, frame):
        s = self.settings['dilate']
        result = cv2.dilate(frame, None, iterations = s)
        return result

    def opening(self, frame):
        s = self.settings['open']
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations = s)
        return opening

    def closing(self, frame):
        s = self.settings['close']
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations = s)
        return closing

    def skeletonize(self, frame):
        # From https://stackoverflow.com/questions/33095476/is-there-any-build-in-function-can-do-skeletonization-in-opencv
        img = frame
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        s = self.settings['skel']
        if s == 0:
            skel = frame
        else:
            skel = np.zeros(img.shape,np.uint8)

        for _ in range(s):
            eroded = cv2.erode(img,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(img,temp)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()

        return skel

    def get_frame(self):
        if self.source_type == 'cap':
            ret, self.frame = self.cap.read()
            return self.frame
        if self.source_type == 'ros':
            return self.frame
        if self.source_type == 'image':
            return self.frame


