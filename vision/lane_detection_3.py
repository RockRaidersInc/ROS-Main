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
    cv2_img = None

    def __init__(self):
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


    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Or use RGB2GRAY if you read an image with mpimg
    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def x_thresh(self, img, sobel_kernel=3, thresh=(0, 255)):
        gray = self.grayscale(img)
        # Take only Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        # Calculate the absolute value of the x derivative:
        abs_sobelx = np.absolute(sobelx)
        # Convert the absolute value image to 8-bit:
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        # Create binary image using thresholding
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return sxbinary

    def edge_thresh(self, img, thresh=(0, 255)):
        gray = self.grayscale(img)
        # Take the scharr filter in x and y. Scharr is like sobel but has better rotational symmetry
        # see for details - https://en.wikipedia.org/wiki/Sobel_operator#Alternative_operators
        scharrx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=-1)
        scharry = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=-1)
        # Calculate the absolute value of the x derivative:
        scharr_mag = np.sqrt(np.power(scharrx, 2) + np.power(scharry, 2))
        # Convert the absolute value image to 8-bit:
        scaled_scharr = np.uint8(255 * scharr_mag / np.max(scharr_mag))
        # Create binary image using thresholding
        sxybinary = np.zeros_like(scaled_scharr)
        sxybinary[(scaled_scharr >= thresh[0]) & (scaled_scharr <= thresh[1])] = 1
        return sxybinary

    def mag_thresh(self, img, sobel_kernel=3, thresh=(0, 255)):
        # Convert to grayscale
        gray = self.grayscale(img)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

        # Return the binary image
        return binary_output

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

        # src = np.float32([[130, 310], [231, 172], [431, 173], [563, 309]])
        # img_x_half = 1280 / 2
        # img_y_half = 720 / 2
        img_x_half = img.shape[1] / 2
        img_y_half = img.shape[0] / 2
        y_offset = 720 / 2 - 80
        square_size_y = 0.508 * 100
        square_size_x = 0.762 * 100
        # dst = np.float32([[-square_size/2 + img_x_half, square_size/2 + img_y_half + y_offset], 
        #                 [-square_size/2 + img_x_half, -square_size/2 + img_y_half + y_offset], 
        #                 [square_size/2 + img_x_half, -square_size/2 + img_y_half + y_offset], 
        #                 [square_size/2 + img_x_half, square_size/2 + img_y_half + y_offset]])

        # src = np.float32([[523, 491], [555, 383], [779, 395], [786, 501]])
        # dst = np.float32([[-square_size_x/2 + img_x_half, square_size_y/2 + img_y_half + y_offset], 
        #                 [-square_size_x/2 + img_x_half, -square_size_y/2 + img_y_half + y_offset],
        #                 [square_size_x/2 + img_x_half, -square_size_y/2 + img_y_half + y_offset],
        #                 [square_size_x/2 + img_x_half, square_size_y/2 + img_y_half + y_offset]])
        

        """
        good calibration: (0ft, 0ft right inbetween wheels, points in x,y format where rover drives in y direction)
            bottom: 701, 653 - 0ft, 3ft
            center: 691, 457 - 0ft, 5ft
            top: 658, 327 - 0ft, 7ft
            left: 471, 456 - -2ft, 5ft
            right: 920, 457 - 2ft, 5ft
        """
        y_offset = 720 / 2 - 80
        scale_factor = 50
        src = np.float32([[701, 653], 
                        #  [691, 457], 
                         [658, 327], 
                         [471, 456], 
                         [920, 457]])
        dst = np.float32([[0 * scale_factor + img_x_half, 2 * scale_factor + img_y_half + y_offset],  # bottom
                        # [0 * scale_factor + img_x_half, 0 * scale_factor + img_y_half + y_offset],    # center
                        [0 * scale_factor + img_x_half, -2 * scale_factor + img_y_half + y_offset],   # top
                        [-2 * scale_factor + img_x_half, 0 * scale_factor + img_y_half + y_offset],   # left
                        [2 * scale_factor + img_x_half, 0 * scale_factor + img_y_half + y_offset]])   # right


        # print "src:", src
        # print "dst:", dst

        M = cv2.getPerspectiveTransform(src, dst)

        # inverse
        Minv = cv2.getPerspectiveTransform(dst, src)

        # create a warped image
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        roi_unwarped = np.array([[[0, 0]], [[0, img_y_half * 2]], [[img_x_half * 2, img_y_half * 2]], [[img_x_half * 2, 0]]], dtype="float32")
        roi_warped = cv2.perspectiveTransform(roi_unwarped, M)

        # unpersp = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_LINEAR)
        unpersp = img

        return warped, roi_warped, Minv


    def warp_points(self, point_array, img_shape):
        img_size = img_shape

        """
        good calibration: (0ft, 0ft right inbetween wheels, points in x,y format where rover drives in y direction)
            bottom: 701, 653 - 0ft, 3ft
            center: 691, 457 - 0ft, 5ft
            top: 658, 327 - 0ft, 7ft
            left: 471, 456 - -2ft, 5ft
            right: 920, 457 - 2ft, 5ft
        """

        img_x_half = img_shape[1] / 2
        img_y_half = img_shape[0] / 2
        y_offset = 720 / 2 - 80

        y_offset = 720 / 2 - 80
        scale_factor = 50
        src = np.float32([[701, 653], 
                        #  [691, 457], 
                         [658, 327], 
                         [471, 456], 
                         [920, 457]])
        dst = np.float32([[0.9144, 0],  # bottom
                        # [0 * scale_factor + img_x_half, 0 * scale_factor + img_y_half + y_offset],    # center
                        [2.1336, 0],   # top
                        [1.524, -0.6096],   # left
                        [1.524, 0.6096]])   # right

        M = cv2.getPerspectiveTransform(src, dst)

        roi_unwarped = np.array([[[0, 0]], [[0, img_y_half * 2]], [[img_x_half * 2, img_y_half * 2]], [[img_x_half * 2, 0]]], dtype=np.float32)
        roi_warped = cv2.perspectiveTransform(roi_unwarped, M)

        point_array = point_array.transpose()
        reshaped_point_array = point_array.reshape([point_array.shape[0], 1, point_array.shape[1]]).astype(np.float32)
        warped_points = cv2.perspectiveTransform(reshaped_point_array, M)

        return warped_points, roi_warped

    # Function for saving images to an output folder
    def create_pathname(self, infile, ext):
        temp1 = os.path.split(infile)[-1]
        temp2 = os.path.splitext(temp1)[0] + ext
        outfile = os.path.join("output1/", temp2)
        return outfile


    def u_pts_pub_test(self, roi, points):
        try:
            roi = np.array(roi).squeeze()
            points = np.array(points).squeeze()
            print points.shape
            lane = Lane()
            
            tl_bound_pt = Vector3(np.max(roi[:, 1]), np.max(roi[:, 0]), 0.0)
            br_bound_pt = Vector3(np.min(roi[:, 1]), np.min(roi[:, 0]), 0.0)
            # tl_bound_pt = Vector3(1.1,1.1,0.0)
            # br_bound_pt = Vector3(-.1,-1.1,0.0)
            lane.bound_corners[0] = tl_bound_pt
            lane.bound_corners[1] = br_bound_pt

            # import matplotlib.pyplot as plt
            # plt.scatter(points[:, 0], points[:, 1])
            # plt.show()
            # plt.clear()

            # for i in range(10):
            #     lane_pt = Vector3(i/10.0,1.0,0.0)	
            #     lane.lane_points.append(lane_pt)
            for i in range(points.shape[0]):
                lane_pt = Vector3(points[i, 0], points[i, 1], 0.0)	
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
    def lane_detector(self, image, video_mode=False):
        # Undistort image
        # undist, roi, Minv = self.warp(image)
        undist = image

        # Define a kernel size and apply Gaussian smoothing
        apply_blur = True
        if apply_blur:
            undist = self.gaussian_blur(undist, 5)
            extra_blurred = self.gaussian_blur(undist, 21)
        self.blurred_pub.publish(bridge.cv2_to_imgmsg((undist).astype(np.uint8)))

        warped, _, _ = self.warp(undist)
        self.undist_pub.publish(bridge.cv2_to_imgmsg(warped))

        # # Define parameters for gradient thresholding
        # sxybinary = self.edge_thresh(undist, thresh=(22, 100))
        # self.sxybinary_pub.publish(bridge.cv2_to_imgmsg((sxybinary*255).astype(np.uint8)))

        # mag_binary = self.mag_thresh(undist, sobel_kernel=3, thresh=(40, 100))
        # self.mag_binary_pub.publish(bridge.cv2_to_imgmsg((mag_binary*255).astype(np.uint8)))
        # # dir_binary = self.dir_threshold(undist, sobel_kernel=15, thresh=(0.7, 1.3))

        # Define parameters for color thresholding
        # s_binary = self.hls_select(undist, thresh=(90, 255))
        s_binary = self.hsv_select(undist)
        self.s_binary_pub.publish(bridge.cv2_to_imgmsg(s_binary.astype(np.uint8)))

        # You can combine various thresholding operations

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        # color_binary = np.dstack((np.zeros_like(sxybinary), sxybinary, s_binary))

        # Combine the two binary thresholds
        # combined_binary1 = np.zeros_like(sxybinary)
        # combined_binary1[(s_binary == 1) | (sxybinary == 1)] = 1
        # self.combined_binary1_pub.publish(bridge.cv2_to_imgmsg(combined_binary1*255))

        # do a skeleton 
        # bool done;
        # do
        # {
        # cv::morphologyEx(img, temp, cv::MORPH_OPEN, element);
        # cv::bitwise_not(temp, temp);
        # cv::bitwise_and(img, temp, temp);
        # cv::bitwise_or(skel, temp, skel);
        # cv::erode(img, img, element);
        
        # double max;
        # cv::minMaxLoc(img, 0, &max);
        # done = (max == 0);
        # } while (!done);

        # combined_binary2 = np.zeros_like(sxybinary)
        # combined_binary2[(s_binary == 1) | (sxybinary == 1) | (mag_binary == 1)] = 1

        # close the image to get rid of noise
        kernel = np.ones((5,5), np.uint8) 
        # eroded = cv2.erode(warped_im, kernel, iterations=1)
        denoised = cv2.morphologyEx(s_binary.astype(np.uint8), cv2.MORPH_ERODE, kernel)
        denoised = self.skeleton(denoised)

        self.denoised_pub.publish(bridge.cv2_to_imgmsg(denoised))

        # Apply perspective transform
        warped_im, roi, Minv = self.warp(np.dstack([denoised, denoised, denoised]))

        # publish points
        # lane_points_image_frame = np.array(np.where(denoised != 0))
        lane_points_image_frame = np.array([[485, 199], [326, 438], [1171, 442]]).transpose()
        lane_points_map_frame, roi = self.warp_points(lane_points_image_frame, denoised.shape)
        print lane_points_map_frame
        self.u_pts_pub_test(roi, lane_points_map_frame)

        # publish the warped image for debugging purposes
        for i in range(len(roi) - 1):
            cv2.line(warped_im, tuple(roi[i, 0]), tuple(roi[i+1, 0]), (0, 255, 0), 4)

        self.warped_im_pub.publish(bridge.cv2_to_imgmsg(warped_im))

        # return s_binary, combined_binary1, warped_im, Minv


    def image_callback(self, msg):
        print("Received an image!")
        try:
            # Convert your ROS Image message to OpenCV2
            self.cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")

        except CvBridgeError as e:
            print("error recieving image")
            print(e)


    def main(self):
        rospy.init_node('image_listener')
        # Define your image topic
        image_topic = "/zed_node/left/image_rect_color"
        # image_topic = "/zed/depth/image_raw"
        # Set up your subscriber and define its callback
        rospy.Subscriber(image_topic, Image, self.image_callback)
        self.raw_pub = rospy.Publisher("raw_image", Image, queue_size=10)
        self.blurred_pub = rospy.Publisher("blurred", Image, queue_size=10)
        self.undist_pub = rospy.Publisher("undist", Image, queue_size=10)
        self.sxybinary_pub = rospy.Publisher("sxybinary", Image, queue_size=10)
        self.s_binary_pub = rospy.Publisher("s_binary", Image, queue_size=10)
        self.combined_binary1_pub = rospy.Publisher("combined_binary1", Image, queue_size=10)
        self.warped_im_pub = rospy.Publisher("warped_im", Image, queue_size=10)
        self.lane_drawn_pub = rospy.Publisher("lane_drawn", Image, queue_size=10)
        self.mag_binary_pub = rospy.Publisher("mag_binary", Image, queue_size=10)
        self.denoised_pub = rospy.Publisher("denoised", Image, queue_size=10)

        self.lane_pub = rospy.Publisher('/lanes', Lane, queue_size=10)
        

        self.debug = rospy.Publisher("detector_debug", Image, queue_size=10)

        # time.sleep(3)
        # Spin until ctrl + c
        while not rospy.is_shutdown():
            if self.cv2_img is not None:
                cv2_img_local = self.cv2_img
                self.raw_pub.publish(bridge.cv2_to_imgmsg(cv2_img_local))
                self.cv2_img = None
                self.lane_detector(cv2_img_local)

            time.sleep(0.01)


if __name__ == '__main__':
    detector = LaneDetector()
    detector.main()

