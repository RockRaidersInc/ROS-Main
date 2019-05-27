#! /usr/bin/python

import numpy as np
import cv2
import pickle
import time
import sys
import os
import yaml

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from autonomous_traversal.msg import Lane
from geometry_msgs.msg import Vector3

from debug_utils import *


bridge = CvBridge()


class LaneDetector:
    # # 720p
    # x_resolution = 1280.0
    # y_resolution = 720.0

    # 480p
    x_resolution = 640.0
    y_resolution = 480.0

    post_filtering_scaling_factor = 4

    max_x_dist = 3.0  # maximum distance away from the rover at which lanes will be detected

    debug = True
    print_timing_info = True

    depth_img = None
    depth_img_time = None

    def __init__(self):
        self.cv2_img = None
        # TODO: Make this file name be inputed as a option
        self.setting_filename = 'settings/trackbar_settings.yaml'
        with open(self.setting_filename, 'r') as file:
            self.settings = yaml.load(file)[int(sys.argv[1])]
            print('Loading image processing settings')
            print(yaml.dump(self.settings, default_flow_style=False))

    def hsv_clr_filter(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_hsv = np.array([self.settings['hl'],self.settings['sl'],self.settings['vl']])
        upper_hsv = np.array([self.settings['hh'],self.settings['sh'],self.settings['vh']])
        clr_filt_mask = cv2.inRange(hsv_img,lower_hsv,upper_hsv)
        
        return clr_filt_mask

    def warp(self, img):
        img_x_half = self.x_resolution / 2
        img_y_half = self.y_resolution / 2
        y_offset = self.y_resolution / 2 - 100
        scale_factor = 25

        src = np.float32([[193 *self.x_resolution/640.0, 426 *self.y_resolution/480.0],  # bottom left
                         [256 *self.x_resolution/640.0, 233 *self.y_resolution/480.0],  # top left
                         [430 *self.x_resolution/640.0, 232 *self.y_resolution/480.0],  # top right
                         [510 *self.x_resolution/640.0, 436 *self.y_resolution/480.0]]) # bottom right

        dst = np.float32([[-2 * scale_factor + img_x_half, 2 * scale_factor + img_y_half + y_offset],   # bottom left
                          [-2 * scale_factor + img_x_half, -2 * scale_factor + img_y_half + y_offset],   # top left
                          [2 * scale_factor + img_x_half, -2 * scale_factor + img_y_half + y_offset],    # top right
                          [2 * scale_factor + img_x_half, 2 * scale_factor + img_y_half + y_offset]])  # bottom right

        M = cv2.getPerspectiveTransform(src, dst)

        # inverse
        M_inv = cv2.getPerspectiveTransform(dst, src)

        # create a warped image
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        roi_unwarped = np.array([[[0, self.y_resolution]], [[0, 0]], [[self.x_resolution, 0]], [[self.x_resolution, self.y_resolution]]], dtype=np.float32)
        roi_warped = cv2.perspectiveTransform(roi_unwarped, M)

        # unpersp = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_LINEAR)
        unpersp = img

        return warped, roi_warped, M_inv

    def warp_points(self, point_array, img_shape):

        src = np.float32([[193 *self.x_resolution/640.0, 426 *self.y_resolution/480.0],  # bottom left
                         [256 *self.x_resolution/640.0, 233 *self.y_resolution/480.0],   # top left
                         [430 *self.x_resolution/640.0, 232 *self.y_resolution/480.0],   # top right
                         [510 *self.x_resolution/640.0, 436 *self.y_resolution/480.0]])  # bottom right
        dst = np.float32([[0.9144, 0.6096],  # bottom left
                        [2.1336, 0.6096],    # top left
                        [2.1336, -0.6096],   # top right
                        [0.9144, -0.6096]])  # bottom right

        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        pt = np.array([[[self.max_x_dist, 0]]], dtype=np.float32)
        max_x_dist_img = cv2.perspectiveTransform(pt, M_inv)

        roi_unwarped = np.array([[[0, self.y_resolution]],                   # bottom left
                                 [[0, max_x_dist_img[0, 0, 1]]],                      # top left
                                 [[self.x_resolution, max_x_dist_img[0, 0, 1]]],      # top right
                                 [[self.x_resolution, self.y_resolution]]])  # bottom right
        roi_warped = cv2.perspectiveTransform(roi_unwarped, M)

        for i in range(roi_warped.shape[0]):
            if roi_warped[i, 0, 0] > self.max_x_dist:
                roi_warped[i, 0, 0] = self.max_x_dist

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

            for point in roi:
                lane.bound_polygon.append(Vector3(point[0], point[1], 0))

            points_filtered_x = []
            points_filtered_y = []
            for i in range(points.shape[0]):
                point_x = points[i, 0]
                point_y = points[i, 1]
                if self.max_x_dist < point_x:
                    # if True:
                    # print(point_x, point_y)
                    points_filtered_x.append(point_x)
                    points_filtered_y.append(point_y)

            # for i in range(len(points_filtered_y)):
                # point_x = points_filtered_x[i]
                # point_y = points_filtered_y[i]

            for i in range(points.shape[0]):
                point_x = points[i, 0]
                point_y = points[i, 1]
                if self.max_x_dist > point_x:
                    lane_pt = Vector3(point_x, point_y, 0.0)
                    lane.lane_points.append(lane_pt)
                else:
                    pass
                    x = 1

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
            print('image resizing took', time.time() - prev_time, 'seconds')
            prev_time = time.time()

        ## Image processing
        # HSV Filter
        hsv_filtered = self.hsv_clr_filter(image)
        if self.print_timing_info:
            print('hsv filtering took', time.time() - prev_time, 'seconds')
            prev_time = time.time()
        # Open to filter out noise
        kernel = np.ones((2,2),np.uint8)
        hsv_filtered = cv2.morphologyEx(hsv_filtered, cv2.MORPH_OPEN, kernel)
        if self.print_timing_info:
            print('Opening took', time.time() - prev_time, 'seconds')
            prev_time = time.time()
        # Make the image smaller by a factor of post_filtering_scaling_factor
        resize_dim = (int(self.x_resolution / self.post_filtering_scaling_factor),
                      int(self.y_resolution / self.post_filtering_scaling_factor))
        hsv_filtered = cv2.resize(hsv_filtered, resize_dim)
        if self.print_timing_info:
            print('resizing filtered image took', time.time() - prev_time, 'seconds')
            prev_time = time.time()
        # Skeletonize mask to reduce number of lane points
        skeletoned = self.skeleton(hsv_filtered.astype(np.uint8))
        if self.print_timing_info:
            print('skeleton took', time.time() - prev_time, 'seconds')
            prev_time = time.time()
        # Calculate lane points and Warp perspective
        lane_points_image_frame_yx = np.array(np.where(skeletoned != 0)) * self.post_filtering_scaling_factor
        lane_points_image_frame = np.array([lane_points_image_frame_yx[1,:], lane_points_image_frame_yx[0,:]])
        lane_points_map_frame, point_roi = self.warp_points(lane_points_image_frame, skeletoned.shape)
        if self.print_timing_info:
            print('selecting points took', time.time() - prev_time, 'seconds')
            prev_time = time.time()

        if self.debug:
            warped_im, _, Minv = self.warp(np.dstack([skeletoned, skeletoned, skeletoned]))
            self.warped_im_pub.publish(bridge.cv2_to_imgmsg(skeletoned))

            if self.print_timing_info:
                print('warping and publishing input image took', time.time() - prev_time, 'seconds')
                prev_time = time.time()

        # Publish points and image
        self.publish_lane_pts(point_roi, lane_points_map_frame)

        if self.print_timing_info:
            print('publishing points and data took', time.time() - prev_time, 'seconds')
            print('took', time.time() - start_time, 'seconds in total')
            print

        if self.debug:
            # publish the warped image for debugging purposes
            warped_im_raw, roi, Minv = self.warp(image)
            for i in range(len(roi) - 1):
                cv2.line(warped_im_raw, tuple(roi[i, 0]), tuple(roi[i + 1, 0]), (0, 255, 0), 4)
            self.debug_img_pub.publish(bridge.cv2_to_imgmsg(warped_im_raw))

            # publish resized input image
            self.resized_pub.publish(bridge.cv2_to_imgmsg(image.astype(np.uint8), 'bgr8'))
            self.color_filtered_pub.publish((bridge.cv2_to_imgmsg(np.dstack([hsv_filtered, hsv_filtered, hsv_filtered]).astype(np.uint8), 'bgr8')))

    def image_callback(self, msg):
        print('Received an image!')
        try:
            self.cv2_img, self.cv2_img_time = bridge.imgmsg_to_cv2(msg, "bgr8"), msg.header.stamp
            self.lane_detector(self.cv2_img)
        except CvBridgeError as e:
            print('error recieving image\n', e)

    def depth_image_callback(self, msg):
        print("recieved depth image")
        try:
            depth_img = cv2.resize(bridge.imgmsg_to_cv2(msg), (int(self.x_resolution), int(self.y_resolution)))
            self.depth_img, self.depth_image_time = depth_img, msg.header.stamp

        except CvBridgeError as e:
            print("error receiving depth image\n", e)


    def main(self):
        rospy.init_node('image_listener')
        # image_topic = "/zed/image/image_raw"
        image_topic = "/zed_node/left/image_rect_color"
        rospy.Subscriber(image_topic, Image, self.image_callback)

        depth_image_topic = "/zed_node/depth/depth_registered"
        rospy.Subscriber(depth_image_topic, Image, self.depth_image_callback)

        self.raw_pub = rospy.Publisher("raw_image", Image, queue_size=10)
        self.warped_im_pub = rospy.Publisher("warped_im", Image, queue_size=10)
        self.debug_img_pub = rospy.Publisher('/debug_img', Image, queue_size=10)
        self.lane_pub = rospy.Publisher('/lanes', Lane, queue_size=10)
        self.resized_pub = rospy.Publisher('/resized_input', Image, queue_size=10)
        self.color_filtered_pub = rospy.Publisher('/color_filtered', Image, queue_size=10)

        while not rospy.is_shutdown():
            # if self.cv2_img is not None:
            #     self.lane_detector(self.cv2_img)
            #     self.cv2_img = None
            time.sleep(0.25)


if __name__ == '__main__':
    detector = LaneDetector()
    detector.main()
