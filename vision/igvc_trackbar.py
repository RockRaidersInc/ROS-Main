# DEPRECATED, USE igvc_processing.py

import numpy as np
import cv2
import sys
import yaml

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


bridge = CvBridge()


def nothing(x):
    pass

class Trackbar:
    SETTING_FILENAME = 'settings/trackbar_settings.yaml'

    def __init__(self, source_type, source):
        self.source_type = source_type
        self.source = source

        if source_type == 'image':
            pass
        if source_type == 'ros':
            self.frame = None
        if source_type == 'video':
            pass
        if source_type == 'cap':
            pass

        self.settings = {}

    def createSettingsWindow(self):
        print('Creating settings window')
        cv2.namedWindow("Settings",flags=cv2.WINDOW_NORMAL)
        # Add trackbars here
        cv2.createTrackbar('H_low','Settings',0,255,nothing)
        cv2.createTrackbar('H_high','Settings',255,255,nothing)
        cv2.createTrackbar('S_low','Settings',0,255,nothing)
        cv2.createTrackbar('S_high','Settings',255,255,nothing)
        cv2.createTrackbar('V_low','Settings',0,255,nothing)
        cv2.createTrackbar('V_high','Settings',255,255,nothing)
        cv2.createTrackbar('open_k_size','Settings',0,10,nothing)

    def save_settings(self):
        with open(self.SETTING_FILENAME, 'a') as file:
            print('Saving settings to file {}'.format(self.SETTING_FILENAME))
            print(yaml.dump(self.settings, default_flow_style=False))
            yaml.dump([self.settings], file)

    def hsv_color_filter(self, frame):
        self.settings['hl'] = cv2.getTrackbarPos("H_low","Settings")
        self.settings['sl'] = cv2.getTrackbarPos('S_low','Settings')
        self.settings['vl'] = cv2.getTrackbarPos('V_low','Settings')
        self.settings['hh'] = cv2.getTrackbarPos('H_high','Settings')
        self.settings['sh'] = cv2.getTrackbarPos('S_high','Settings')
        self.settings['vh'] = cv2.getTrackbarPos('V_high','Settings')

        lower_hsv = np.array([self.settings['hl'],self.settings['sl'],self.settings['vl']])
        upper_hsv = np.array([self.settings['hh'],self.settings['sh'],self.settings['vh']])
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result = cv2.inRange(result,lower_hsv,upper_hsv)

        return result

    def open(self, frame):
        self.settings['open_k_size'] = cv2.getTrackbarPos('open_k_size', 'Settings')
        kernel = np.ones((self.settings['open_k_size'], self.settings['open_k_size']), np.uint8)
        opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('opened', opened)
        return opened
 
    def skeleton(self, frame):
        # from stackoverflow user Dan Masek (non askii character was replaced with s)
        # https://stackoverflow.com/questions/42845747/optimized-skeleton-function-for-opencv-with-python
        skeleton = np.zeros(frame.shape, np.uint8)
        eroded = np.zeros(frame.shape, np.uint8)
        temp = np.zeros(frame.shape, np.uint8)

        _, thresh = cv2.threshold(frame, 127, 255, 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        iters = 0
        while (True):
            cv2.erode(thresh, kernel, eroded)
            cv2.dilate(eroded, kernel, temp)
            cv2.subtract(thresh, temp, temp)
            cv2.bitwise_or(skeleton, temp, skeleton)
            thresh, eroded = eroded, thresh  # Swap instead of copy

            iters += 1
            if cv2.countNonZero(thresh) == 0:
                return skeleton

    def proc_frame(self, frame):
        result = self.hsv_color_filter(frame)
        result = self.open(result)
        # result = self.skeleton(result)
        return result

    def ros_image_callback(self, msg):
        try:
            self.frame = cv2.resize(bridge.imgmsg_to_cv2(msg, "bgr8"), (640, 480))
        except CvBridgeError as e:
            print("error recieving image\n", e)

    def run_image(self):
        pass

    def run_ros(self):
        rospy.init_node('trackbar')
        print('Subscribing to topic: {}'.format(self.source))
        rospy.Subscriber(self.source, Image, self.ros_image_callback)

        while not rospy.is_shutdown():
            if self.frame is not None:
                result = self.proc_frame(self.frame)
                cv2.imshow('Original',self.frame)
                cv2.imshow('Result',result)
                self.frame = None
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                self.save_settings()
            if key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    def run_video(self):
        pass

    def run_cap(self):
        pass

    def run(self):
        self.createSettingsWindow()

        if self.source_type == 'image':
            self.run_image()
        if self.source_type == 'ros':
            self.run_ros()
        if self.source_type == 'video':
            self.run_video()
        if self.source_type == 'cap':
            self.run_cap()


def main():
    source_type = 'ros'
    source = '/zed_node/left/image_rect_color'
    trackbar = Trackbar(source_type, source)
    trackbar.run()

if __name__ == '__main__':
    main()



