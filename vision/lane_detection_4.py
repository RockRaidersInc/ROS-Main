#! /usr/bin/python

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
from debug_utils import *
import os

from sklearn import svm
import csv

import rospy
from autonomous_traversal.msg import Lane
from geometry_msgs.msg import Vector3


bridge = CvBridge()

class LaneDetector:

    # # 720p
    # x_resolution = 1280.0
    # y_resolution = 720.0

    # 640p
    x_resolution = 640.0
    y_resolution = 480.0

    debug = True
    print_timing_info = False

    def __init__(self):
        self.cv2_img = None
        # train svm
        with open("hsv_colors_labeled.csv") as hsv_file:
            readCSV = csv.reader(hsv_file, delimiter=',')
            h = []
            s = []
            v = []
            line = []
            next(readCSV)  # skip the header line
            for row in readCSV:
                h.append(float(row[0]))
                s.append(float(row[1]))
                v.append(float(row[2]))
                line.append(1 if int(row[3]) == 1 else -1)
            
            train_features = np.array([h, s, v]).transpose()
            self.hsv_mean = train_features.mean(axis=0)
            train_features -= self.hsv_mean
            self.hsv_stdev = np.stdev = np.std(train_features, axis=0)
            train_features /= self.hsv_stdev


            class_labels = np.array(line)
            self.clf = svm.SVC(kernel='linear', C = 1.0)
            self.clf.fit(train_features, class_labels)

            # predict the training set as a sanity check
            correct = 0
            for i in range(train_features.shape[0]):
                if self.clf.predict(train_features[i].reshape([1, 3])) == class_labels[i]:
                    correct += 1
            correct_percent = float(correct) / train_features.shape[0]

            self.SVM_normal_vect = np.array(self.clf.coef_).squeeze()
            self.SVM_intercept = self.clf.intercept_

            # predict the training set as a sanity check
            correct = 0
            for i in range(train_features.shape[0]):
                feature = train_features[i]
                prediction = np.dot(feature, self.SVM_normal_vect) + self.SVM_intercept
                if (1 if prediction > 0 else -1) == class_labels[i]:
                    correct += 1
            if float(correct) / train_features.shape[0] != correct_percent:
                print("SVM did not get entire training set correct. This means the SVM is probably not working properly")
                print("got " + str(correct) + " right out of " + str(train_features.shape[0]) + " training points")

    def hsv_select(self, img):
        # use an SVM to select colors that are probably lane
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        data_prepped = (hsv_img - self.hsv_mean) / self.hsv_stdev

        line_vals = np.dot(hsv_img, self.SVM_normal_vect) + self.SVM_intercept
        color_binary = np.where(line_vals > 125, 255, 0)
        
        # line_scores = self.clf.predict(hsv_img.reshape([-1, 3])).reshape(hsv_img.shape[:2])
        # color_binary = (line_scores + 1) * (255. / 2)
        return color_binary

    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])

        img_x_half = img.shape[1] / 2
        img_y_half = img.shape[0] / 2
        y_offset = self.y_resolution / 2 - 80*self.y_resolution/720.0
        square_size_y = 0.508 * 100
        square_size_x = 0.762 * 100
        scale_factor = 50

        src = np.float32([[self.x_resolution - 387*self.x_resolution/1280.0, self.y_resolution - 81*self.y_resolution/720.0],  # bottom left
                         [self.x_resolution - 515*self.x_resolution/1280.0, self.y_resolution - 370*self.y_resolution/720.0],  # top left
                         [self.x_resolution - 865*self.x_resolution/1280.0, self.y_resolution - 371*self.y_resolution/720.0],  # top right
                         [self.x_resolution - 1020*self.x_resolution/1280.0, self.y_resolution - 65*self.y_resolution/720.0]]) # bottom right

        dst = np.float32([[2 * scale_factor + img_x_half, 2 * scale_factor + img_y_half + y_offset],   # bottom left
                          [2 * scale_factor + img_x_half, -2 * scale_factor + img_y_half + y_offset],   # top left
                          [-2 * scale_factor + img_x_half, -2 * scale_factor + img_y_half + y_offset],    # top right
                          [-2 * scale_factor + img_x_half, 2 * scale_factor + img_y_half + y_offset]])  # bottom right

        M = cv2.getPerspectiveTransform(src, dst)

        # inverse
        Minv = cv2.getPerspectiveTransform(dst, src)

        # create a warped image
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        roi_unwarped = np.array([[[0, 0]], [[0, img_size[0]]], [[img_size[1], img_size[0]]], [[img_size[1], 0]]], dtype=np.float32)
        roi_warped = cv2.perspectiveTransform(roi_unwarped, M)

        # unpersp = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_LINEAR)
        unpersp = img

        return warped, roi_warped, Minv

    def warp_points(self, point_array, img_shape):
        img_size = [self.y_resolution, self.x_resolution]

        img_x_half = img_shape[1] / 2
        img_y_half = img_shape[0] / 2
        y_offset = self.y_resolution / 2 - 80*self.y_resolution/720.0
        scale_factor = 50

        src = np.float32([[self.x_resolution - 387*self.x_resolution/1280.0, self.y_resolution - 81*self.y_resolution/720.0],  # bottom left
                         [self.x_resolution - 515*self.x_resolution/1280.0, self.y_resolution - 370*self.y_resolution/720.0],  # top left
                         [self.x_resolution - 865*self.x_resolution/1280.0, self.y_resolution - 371*self.y_resolution/720.0],  # top right
                         [self.x_resolution - 1020*self.x_resolution/1280.0, self.y_resolution - 65*self.y_resolution/720.0]]) # bottom right
        dst = np.float32([[0.9144, -0.6096],     # bottom left
                        [2.1336, -0.6096],       # top left
                        [2.1336, 0.6096],  # top right
                        [0.9144, 0.6096]])  # bottom right

        M = cv2.getPerspectiveTransform(src, dst)

        roi_unwarped = np.array([[[img_size[1]*.1, img_size[0]*.1]], 
                                [[ img_size[1]*.1, img_size[0]*.9]], 
                                [[ img_size[1]*.9, img_size[0]*.9]], 
                                [[ img_size[1]*.9, img_size[0]*.1]]], dtype=np.float32)
        roi_warped = cv2.perspectiveTransform(roi_unwarped, M)

        point_array = point_array.transpose()
        reshaped_point_array = point_array.reshape([point_array.shape[0], 1, point_array.shape[1]]).astype(np.float32)
        warped_points = cv2.perspectiveTransform(reshaped_point_array, M)

        return warped_points, roi_warped

    def publish_lane_pts(self, roi, points):
        try:
            roi = np.array(roi).squeeze()
            points = np.array(points).squeeze()
            # print points.shape
            lane = Lane()

            lower_x_bound = 1.9144
            upper_x_bound = 5.
            lower_y_bound = -2.
            upper_y_bound = 2.

            lane.bound_polygon.append(Vector3(lower_x_bound,lower_y_bound,0))
            lane.bound_polygon.append(Vector3(lower_x_bound,upper_y_bound,0))
            lane.bound_polygon.append(Vector3(upper_x_bound,upper_y_bound,0))
            lane.bound_polygon.append(Vector3(upper_x_bound,lower_y_bound,0))

            points_filtered_x = []
            points_filtered_y = []
            for i in range(points.shape[0]):
                point_x = points[i, 0]
                point_y = points[i, 1]
                if lower_x_bound < point_x and point_x < upper_x_bound and \
                   lower_y_bound < point_y and point_y < upper_y_bound:
                    # if True:
                    # print(point_x, point_y)
                    points_filtered_x.append(point_x)
                    points_filtered_y.append(point_y)

            for i in range(len(points_filtered_y)):
                point_x = points_filtered_x[i]
                point_y = points_filtered_y[i]
                lane_pt = Vector3(point_x, point_y, 0.0)    
                lane.lane_points.append(lane_pt)

            self.lane_pub.publish(lane)
        except IndexError as e:
            print(e)

    def skeleton(self, img):
        # from stackoverflow user Dan Masek (non askii character was replaced with s)
        # https://stackoverflow.com/questions/42845747/optimized-skeleton-function-for-opencv-with-python
        skeleton = np.zeros(img.shape,np.uint8)
        eroded = np.zeros(img.shape,np.uint8)
        temp = np.zeros(img.shape,np.uint8)

        _,thresh = cv2.threshold(img,127,255,0)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

        iters = 0
        while(True):
            cv2.erode(thresh, kernel, eroded)
            cv2.dilate(eroded, kernel, temp)
            cv2.subtract(thresh, temp, temp)
            cv2.bitwise_or(skeleton, temp, skeleton)
            thresh, eroded = eroded, thresh # Swap instead of copy

            iters += 1
            if cv2.countNonZero(thresh) == 0:
                return skeleton

    # Define a function for creating lane lines
    def lane_detector(self, input_image, video_mode=False):
        if self.print_timing_info:
            start_time = time.time()
            prev_time = time.time()

        image = cv2.resize(input_image, (int(self.x_resolution), int(self.y_resolution)))

        if self.print_timing_info:
            print("image resizing took", time.time() - prev_time, "seconds")
            prev_time = time.time()

        # Image processing
        hsv_filtered = self.hsv_select(image)

        if self.print_timing_info:
            print("hsv filtering took", time.time() - prev_time, "seconds")
            prev_time = time.time()

        skeletoned = self.skeleton(hsv_filtered.astype(np.uint8))

        if self.print_timing_info:
            print("skeleton took", time.time() - prev_time, "seconds")
            prev_time = time.time()

        # Calculate lane points
        lane_points_image_frame_yx = np.array(np.where(skeletoned != 0))
        lane_points_image_frame = np.array([lane_points_image_frame_yx[1,:], lane_points_image_frame_yx[0,:]])
        # Warp perspective
        lane_points_map_frame, roi = self.warp_points(lane_points_image_frame, skeletoned.shape)

        if self.print_timing_info:
            print("selecting points took", time.time() - prev_time, "seconds")
            prev_time = time.time()

        warped_im, roi, Minv = self.warp(np.dstack([skeletoned, skeletoned, skeletoned]))

        if self.print_timing_info:
            print("warping input image took", time.time() - prev_time, "seconds")
            prev_time = time.time()

        # Publish points and image
        self.publish_lane_pts(roi, lane_points_map_frame)
        self.warped_im_pub.publish(bridge.cv2_to_imgmsg(warped_im))

        if self.print_timing_info:
            print("publishing points and data took", time.time() - prev_time, "seconds")
            print("took", time.time() - start_time, "seconds in total")
            print

    def image_callback(self, msg):
        print("Received an image!")
        try:
            # self.cv2_img = cv2.resize(bridge.imgmsg_to_cv2(msg, "bgr8"), (int(self.x_resolution), int(self.y_resolution)))
            self.cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            self.lane_detector(self.cv2_img)
        except CvBridgeError as e:
            print("error recieving image\n", e)


    def main(self):
        rospy.init_node('image_listener')
        # image_topic = "/zed/image/image_raw"
        image_topic = "/zed_node/left/image_rect_color"
        rospy.Subscriber(image_topic, Image, self.image_callback)

        self.raw_pub = rospy.Publisher("raw_image", Image, queue_size=10)
        self.warped_im_pub = rospy.Publisher("warped_im", Image, queue_size=10)
        self.debug_img_pub = rospy.Publisher('/debug_img', Image, queue_size=10)
        self.lane_pub = rospy.Publisher('/lanes', Lane, queue_size=10)

        while not rospy.is_shutdown():
            # if self.cv2_img is not None:
            #     self.lane_detector(self.cv2_img)
            #     self.cv2_img = None

            # turn this delay down if running image detection in main thread
            time.sleep(0.5)


if __name__ == '__main__':
    detector = LaneDetector()
    detector.main()
