#!/usr/bin/env python

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
import glob
import rospy
from sensor_msgs.msg import Image
import sys


class node:
    def __init__(self):
        rospy.init_node("image_saver")
        if len(sys.argv) != 3:
            print("must pass an image file and topic name as an arguments")
            print("usage: rosrun utils static_image_publisher filename.png /topic_name")
            sys.exit(1)

        self.bridge = CvBridge()
        sub = rospy.Subscriber(sys.argv[2], Image, self.callback)

    def callback(self, img_msg):
        print("got image, saving and exiting")
        cv2_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv2.imwrite(sys.argv[1], cv2_img)
        rospy.signal_shutdown("node done")


if __name__ == "__main__":
    node()
    rospy.spin()
