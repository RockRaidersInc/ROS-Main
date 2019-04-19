#!/usr/bin/env python

import rospy
import actionlib
from geonav_transform import geonav_conversions as gc
from move_base_msgs.msg import MoveBaseAction
from general import make_goal


class WaypointNavigator:
	def __init__(self, orig_lat, orig_lon):
		self.orig_lat = orig_lat
		self.orig_lon = orig_lon
		rospy.loginfo('Creating move_base client')
		self.mb_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
		rospy.loginfo('Waiting for server')
		self.mb_client.wait_for_server()

	def send_gps_goal(self, goal_lat, goal_lon):
		self.mb_client.wait_for_server()
		rospy.loginfo('------------------------------------------')
		rospy.loginfo('Making goal with (lat,lon): ({},{})'.format(goal_lat, goal_lon))
		goal_x, goal_y = gc.ll2xy(goal_lat, goal_lon,
								  self.orig_lat, self.orig_lon)
		rospy.loginfo('(lat,lon): ({},{}) converts to \n(x,y): ({},{})'\
					  .format(goal_lat, goal_lon, goal_x, goal_y))
		move_base_goal = make_goal(goal_x, goal_y)
		rospy.loginfo('Sending goal')
		self.mb_client.send_goal(move_base_goal)
		rospy.loginfo('Waiting for results')
		self.mb_client.wait_for_result()