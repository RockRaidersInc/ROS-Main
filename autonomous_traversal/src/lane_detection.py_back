#! /usr/bin/python

import numpy as np
import cv2
import pickle
import time
import sys
import os
import yaml

import rospy
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from autonomous_traversal.msg import Lane
from geometry_msgs.msg import Vector3


bridge = CvBridge()

raw_pub = rospy.Publisher("/raw_image", Image, queue_size=10)
depth_mask_pub = rospy.Publisher('/depth_mask', Image, queue_size=10)
hsv_clr_filter_pub = rospy.Publisher('/hsv_clr_filter', Image, queue_size=10)
warped_im_pub = rospy.Publisher("/warped_im", Image, queue_size=10)
debug_img_pub = rospy.Publisher('/debug_img', Image, queue_size=10)

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
    print_timing_info = False

    def __init__(self):
        self.cv2_img = None
        self.depth_img = None
        #For debugging

        # For lane publishing
        self.lane_pub = rospy.Publisher('/lanes', Lane, queue_size=10)
        # Load settings
        # TODO: Make this file name be inputed as a option
        rospack = rospkg.RosPack()
        AT_file_path = rospack.get_path('autonomous_traversal')
        self.setting_filepath = '{}/settings/lane_detection.yaml'.format(AT_file_path)
        with open(self.setting_filepath, 'r') as file:
            self.settings = yaml.load(file)[int(sys.argv[1])]
            rospy.loginfo('Loading image processing settings')
            rospy.loginfo(yaml.dump(self.settings, default_flow_style=False))

    def img_proc_timeit(img_proc_name):
        def img_proc_timeit_(func):
            def img_proc_func(self, *args, **kwargs):
                t_start = time.time()
                result = func(self, *args, **kwargs)
                t_elapse_ms = (time.time() - t_start) * 1000 
                if self.print_timing_info:
                    rospy.loginfo("{} took : {:.3f} ms".format(img_proc_name, t_elapse_ms))
                return result
            return img_proc_func
        return img_proc_timeit_

    def pub_img(publisher):
        def pub_img_(func):
            def pub_img_func(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                if self.debug:
                    try:
                        publisher.publish(bridge.cv2_to_imgmsg(result),'bgr8')
                    except TypeError as te: # result is a tuple or list 
                        publisher.publish(bridge.cv2_to_imgmsg(result[0]),'bgr8')
            return pub_img_func
        return pub_img_

    @img_proc_timeit('resizing img')
    def resize(self, img, dim):
        image = cv2.resize(img, dim)
        return image

    @img_proc_timeit('average filter')
    def average_filter(self, frame):
        s = self.settings['avg']
        if s < 1:
            result = frame
        else:
            result = cv2.blur(frame, (s, s))
        return result

    @img_proc_timeit('gaussian filter')
    def gaussian_filter(self, frame):
        s = self.settings['gauss']
        if s < 1:
            result = frame
        else:
            result = cv2.GaussianBlur(frame, (s, s), s)
        return result

    @img_proc_timeit('median filter')
    def median_filter(self, frame):
        s = self.settings['med']
        if s < 1:
            result = frame
        else:
            result = cv2.medianBlur(frame, s)
        return result

    # @pub_img(hsv_clr_filter_pub)
    @img_proc_timeit('hsv thresholding')
    def hsv_clr_filter(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_hsv = np.array([self.settings['hl'], self.settings['sl'], self.settings['vl']])
        upper_hsv = np.array([self.settings['hh'], self.settings['sh'], self.settings['vh']])
        clr_filt_mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

        return clr_filt_mask
    
    @img_proc_timeit('morph erode')
    def morph_erode(self, frame):
        k = self.settings['erode_ksize']
        s = self.settings['erode_iter']
        kernel = np.ones((k,k),np.uint8)
        result = cv2.erode(frame, kernel, iterations = s)
        return result

    @img_proc_timeit('morph dilate')
    def morph_dilate(self, frame):
        k = self.settings['dilate_ksize']
        s = self.settings['dilate_iter']
        kernel = np.ones((k,k),np.uint8)
        result = cv2.dilate(frame, kernel, iterations = s)
        return result

    @img_proc_timeit('morph open')
    def morph_open(self, frame):
        k = self.settings['open_ksize']
        kernel = np.ones((k,k),np.uint8)
        opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations = 1)
        return opening

    @img_proc_timeit('morph close')
    def morph_close(self, frame):
        k = self.settings['close_ksize']
        kernel = np.ones((k,k),np.uint8)
        closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations = 1)
        return closing

    # @pub_img(warped_im_pub)
    @img_proc_timeit('warping img')
    def warp(self, img):
        img_x_half = self.x_resolution / 2
        img_y_half = self.y_resolution / 2
        y_offset = self.y_resolution / 2 - 100
        scale_factor = 25

        src = np.float32([[193 * self.x_resolution / 640.0, 426 * self.y_resolution / 480.0],  # bottom left
                          [256 * self.x_resolution / 640.0, 233 * self.y_resolution / 480.0],  # top left
                          [430 * self.x_resolution / 640.0, 232 * self.y_resolution / 480.0],  # top right
                          [510 * self.x_resolution / 640.0, 436 * self.y_resolution / 480.0]])  # bottom right

        dst = np.float32([[-2 * scale_factor + img_x_half, 2 * scale_factor + img_y_half + y_offset],  # bottom left
                          [-2 * scale_factor + img_x_half, -2 * scale_factor + img_y_half + y_offset],  # top left
                          [2 * scale_factor + img_x_half, -2 * scale_factor + img_y_half + y_offset],  # top right
                          [2 * scale_factor + img_x_half, 2 * scale_factor + img_y_half + y_offset]])  # bottom right

        M = cv2.getPerspectiveTransform(src, dst)

        # inverse
        M_inv = cv2.getPerspectiveTransform(dst, src)

        # create a warped image
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        roi_unwarped = np.array(
            [[[0, self.y_resolution]], [[0, 0]], [[self.x_resolution, 0]], [[self.x_resolution, self.y_resolution]]],
            dtype=np.float32)
        roi_warped = cv2.perspectiveTransform(roi_unwarped, M)

        # unpersp = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_LINEAR)
        unpersp = img

        return warped, roi_warped, M_inv

    @img_proc_timeit('warping pts')
    def warp_points(self, point_array, img_shape):


        # igvc calibration
        src = np.float32([[180 * self.x_resolution / 640.0, 463 * self.y_resolution / 480.0],  # bottom left
                          [248 * self.x_resolution / 640.0, 261 * self.y_resolution / 480.0],  # top left
                          [413 * self.x_resolution / 640.0, 258 * self.y_resolution / 480.0],  # top right
                          [499 * self.x_resolution / 640.0, 455 * self.y_resolution / 480.0]])  # bottom right
        dst = np.float32([[0.9144, 0.6096],  # bottom left
                          [2.1336, 0.6096],  # top left
                          [2.1336, -0.6096],  # top right
                          [0.9144, -0.6096]])  # bottom right

        
        # # previous calibration
        # src = np.float32([[193 * self.x_resolution / 640.0, 426 * self.y_resolution / 480.0],  # bottom left
        #                   [256 * self.x_resolution / 640.0, 233 * self.y_resolution / 480.0],  # top left
        #                   [430 * self.x_resolution / 640.0, 232 * self.y_resolution / 480.0],  # top right
        #                   [510 * self.x_resolution / 640.0, 436 * self.y_resolution / 480.0]])  # bottom right
        # dst = np.float32([[0.9144, 0.6096],  # bottom left
        #                   [2.1336, 0.6096],  # top left
        #                   [2.1336, -0.6096],  # top right
        #                   [0.9144, -0.6096]])  # bottom right

        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        pt = np.array([[[self.max_x_dist, 0]]], dtype=np.float32)
        max_x_dist_img = cv2.perspectiveTransform(pt, M_inv)

        roi_unwarped = np.array([[[0, self.y_resolution]],  # bottom left
                                 [[0, max_x_dist_img[0, 0, 1]]],  # top left
                                 [[self.x_resolution, max_x_dist_img[0, 0, 1]]],  # top right
                                 [[self.x_resolution, self.y_resolution]]])  # bottom right
        roi_warped = cv2.perspectiveTransform(roi_unwarped, M)

        for i in range(roi_warped.shape[0]):
            if roi_warped[i, 0, 0] > self.max_x_dist:
                roi_warped[i, 0, 0] = self.max_x_dist

        point_array = point_array.transpose()
        reshaped_point_array = point_array.reshape([point_array.shape[0], 1, point_array.shape[1]]).astype(np.float32)
        warped_points = cv2.perspectiveTransform(reshaped_point_array, M)

        return warped_points, roi_warped

    @img_proc_timeit('publishing lanes')
    def publish_lane_pts(self, roi, points):
        try:
            roi = np.array(roi).squeeze()
            points = np.array(points).squeeze()
            lane = Lane()

            for point in roi:
                lane.bound_polygon.append(Vector3(point[0], point[1], 0))

            points_filtered_x = []
            points_filtered_y = []
            for i in range(points.shape[0]):
                point_x = points[i, 0]
                point_y = points[i, 1]
                if self.max_x_dist < point_x:
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
            rospy.logwarn(e)

    @img_proc_timeit('skeletonization')
    def skeleton(self, img):
        # from stackoverflow user Dan Masek (non askii character was replaced with s)
        # https://stackoverflow.com/questions/42845747/optimized-skeleton-function-for-opencv-with-python
        skeleton = np.zeros(img.shape, np.uint8)
        eroded = np.zeros(img.shape, np.uint8)
        temp = np.zeros(img.shape, np.uint8)

        _, thresh = cv2.threshold(img, 127, 255, 0)

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

    @img_proc_timeit('get obst pts dy')
    def get_obstacle_points_y_deriv2(self, depth_img, output_size, color_img):
        depth_img = cv2.resize(depth_img, output_size)
        blurred = cv2.GaussianBlur(depth_img, (3, 3), 0)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=-1)

        filtered = np.zeros_like(sobely, dtype=np.uint8)
        filtered[sobely > -0.03] = 1
        nan_img = np.isnan(depth_img)
        nan_img_filtered = cv2.morphologyEx(nan_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))

        filtered = filtered | nan_img_filtered

        expanded = 1 - filtered
        expanded = cv2.morphologyEx(expanded, cv2.MORPH_ERODE, np.ones((11, 11), np.uint8))
        # if self.debug:
            # self.depth_mask_pub.publish(bridge.cv2_to_imgmsg(color_img * expanded[:, :, np.newaxis], "bgr8"))
        return expanded

    @img_proc_timeit('lane pts select')
    def select_lane_pts(self, mask):
        lane_points_image_frame_yx = np.array(np.where(mask != 0)) * self.post_filtering_scaling_factor
        # Swap dim from y,x to x,y (img frame and map frame not alighted)
        lane_points_image_frame = np.array([lane_points_image_frame_yx[1, :], lane_points_image_frame_yx[0, :]])
        return lane_points_image_frame

    # Define a function for creating lane lines
    @img_proc_timeit('lane detection')
    def lane_detector(self, input_image, depth_img=None):
        if depth_img is None:
            depth_img = self.depth_img
        # Preprocessing
        resize_dim = (int(self.x_resolution), int(self.y_resolution))
        resized_input = self.resize(input_image, resize_dim)

        if self.debug:
            self.resized_pub.publish(bridge.cv2_to_imgmsg(resized_input.astype(np.uint8), 'bgr8'))


        # TODO: Noise filtering
        blurred = self.average_filter(resized_input)
        gaussed = self.gaussian_filter(blurred)
        medianed = self.median_filter(gaussed)
        # Color thresholding
        hsv_filtered = self.hsv_clr_filter(medianed)
        # TODO: Morphological operations
        eroded = self.morph_erode(hsv_filtered)
        dilated = self.morph_dilate(eroded)
        opened = self.morph_open(dilated)
        closed = self.morph_close(opened)
        # Resizing by scale factor. Shouldn't this happen in the beginning?
        resize_dim = (int(self.x_resolution / self.post_filtering_scaling_factor),
                      int(self.y_resolution / self.post_filtering_scaling_factor))
        hsv_filtered_smaller = self.resize(closed, resize_dim)
        image_smaller = self.resize(resized_input, resize_dim)
        # Apply depth mask
        if depth_img is not None:
            # don't use depth data if it hasn't been recieved
            valid_mask = self.get_obstacle_points_y_deriv2(depth_img, resize_dim, image_smaller)
            masked_img = (hsv_filtered_smaller*valid_mask).astype(np.uint8)
        else:
            rospy.logwarn("WARNING: no depth image recieved")
            return 
        # Select lane points and Warp perspective
        lane_points_image_frame = self.select_lane_pts(masked_img)
        lane_points_map_frame, point_roi = self.warp_points(lane_points_image_frame, masked_img.shape)
        # Publish points and image
        if lane_points_map_frame is not None:
            self.publish_lane_pts(point_roi, lane_points_map_frame)

    def image_callback(self, msg):
        # rospy.loginfo('Received an image!')
        try:
            self.cv2_img, self.cv2_img_time = bridge.imgmsg_to_cv2(msg, "bgr8"), msg.header.stamp
            self.lane_detector(self.cv2_img)
        except CvBridgeError as e:
            rospy.logwarn('error recieving image\n{}'.format(e))
            pass

    def depth_image_callback(self, msg):
        # rospy.loginfo('recieved depth image')
        try:
            depth_img = cv2.resize(bridge.imgmsg_to_cv2(msg), (int(self.x_resolution), int(self.y_resolution)))
            self.depth_img, self.depth_image_time = depth_img, msg.header.stamp
        except CvBridgeError as e:
            rospy.logwarn('error receiving depth image\n{}'.format(e))

    def main(self):
        rospy.init_node('image_listener')

        # image_topic = rospy.get_param('image_topic')
        # depth_image_topic = rospy.get_param('depth_image_topic') 
        # rospy.Subscriber(image_topic, Image, self.image_callback)
        # rospy.loginfo('Subscribed to image topic: {}'.format(image_topic))
        # rospy.Subscriber(depth_image_topic, Image, self.depth_image_callback)
        # rospy.loginfo('Subscribed to depth image topic: {}'.format(depth_image_topic))


        # simulator
        sim_image_topic = "/zed/image/image_raw"
        sim_depth_image_topic = '/zed/depth/image_raw'
        rospy.Subscriber(sim_image_topic, Image, self.image_callback)
        rospy.Subscriber(sim_depth_image_topic, Image, self.depth_image_callback)
        rospy.loginfo('Subscribed to image topic: {}'.format(sim_image_topic))
        rospy.loginfo('Subscribed to depth image topic: {}'.format(sim_depth_image_topic))


        # physical rover
        image_topic = '/zed_node/left/image_rect_color'
        depth_image_topic = '/zed_node/depth/depth_registered'
        rospy.Subscriber(image_topic, Image, self.image_callback)
        rospy.Subscriber(depth_image_topic, Image, self.depth_image_callback)
        rospy.loginfo('Subscribed to image topic: {}'.format(image_topic))
        rospy.loginfo('Subscribed to depth image topic: {}'.format(depth_image_topic))

        self.resized_pub = rospy.Publisher('/resized_input', Image, queue_size=10)

        while not rospy.is_shutdown():
            time.sleep(0.1)


if __name__ == '__main__':
    detector = LaneDetector()
    detector.main()
