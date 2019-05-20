#!/usr/bin/env python

import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg
import rospy
from sensor_msgs.msg import Image
import time
import sys

if __name__ == "__main__":
    rospy.init_node("static_image_publisher")
    if len(sys.argv) != 3:
        print("must pass an image file and topic name as an arguments")
        print("usage: rosrun utils static_image_publisher path_to_image.png /topic_name")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    if image is None:
        print("could not open image " + sys.argv[1])
    pub = rospy.Publisher(sys.argv[2], Image, queue_size=10)

    bridge = CvBridge()

    print("publishing image every second")

    while not rospy.is_shutdown():
        pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))
        time.sleep(1)
