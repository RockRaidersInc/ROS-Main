#!/usr/bin/env python

import rospy
import actionlib
import time
import sys
from sensor_msgs.msg import NavSatFix
from geonav_transform import geonav_conversions as gc
from move_base_msgs.msg import MoveBaseAction
from general import make_goal


class WaypointNavigator:
	def __init__(self, orig_lat, orig_lon, recalc_goals=False, recalc_rate=3.0):
		# Origin for gps datum
		self.orig_lat = orig_lat
		self.orig_lon = orig_lon
		# Whether or not we wish to republish goal positions to compensate 
		# for odometry drift in the event we are not using gps localization
		self.recalc_goals = recalc_goals
		if recalc_goals:
			self.gps_fix = None
			self.gps_filtered = None
			rospy.Subscriber('gps/fix', NavSatFix, self.save_gps_fix_callback)
			rospy.Subscriber('gps/filtered', NavSatFix, self.save_gps_filtered_callback)
		self.recalc_rate = recalc_rate
		# Create move base client
		rospy.loginfo('Creating move_base client')
		self.mb_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
		rospy.loginfo('Waiting for server')
		self.mb_client.wait_for_server()

	def send_gps_goal(self, goal_lat, goal_lon, timeout=0):
		self.mb_client.wait_for_server()
		# Convert gps coordinates to odometry coordinates
		rospy.loginfo('------------------------------------------')
		rospy.loginfo('Making goal with (lat,lon): ({},{})'.format(goal_lat, goal_lon))
		goal_x, goal_y = gc.ll2xy(goal_lat, goal_lon,
								  self.orig_lat, self.orig_lon)
		rospy.loginfo('(lat,lon): ({},{}) converts to \n(x,y): ({},{})'\
					  .format(goal_lat, goal_lon, goal_x, goal_y))
		# With odom drift compensation
		if self.recalc_goals:
			while not rospy.is_shutdown():
				# Check if we have gps readings
				if self.gps_fix is None or self.gps_filtered is None:
					time.sleep(.1)
					continue
				# Send goals while recalculating
				rospy.loginfo('.................................')
				new_goal = self.recalculate_goal(goal_x, goal_y)
				move_base_goal = make_goal(*new_goal)
				self.mb_client.send_goal(move_base_goal)
				state = self.mb_client.wait_for_result(rospy.Duration.from_sec(self.recalc_rate))
				if state == actionlib.SimpleGoalState.DONE:
					rospy.loginfo('Goal successfully reached')
					break
				# elif state == actionlib.SimpleGoalState.PENDING:
				# 	rospy.logwarn('PENDING State, this should only happen if you ctrl-c')
				# 	sys.exit()
		# Normal gps waypoint to waypoint navigation
		else:
			move_base_goal = make_goal(goal_x, goal_y)
			rospy.loginfo('Sending goal')
			self.mb_client.send_goal(move_base_goal)
			rospy.loginfo('Waiting for results')
			success = self.mb_client.wait_for_result(rospy.Duration.from_sec(timeout))
			if success:
				rospy.loginfo('Goal successfully reached')
			else:
				rospy.logwarn('Was not able to reach the goal in time')

	def save_gps_fix_callback(self, gps_msg):
		self.gps_fix = (gps_msg.latitude, gps_msg.longitude)

	def save_gps_filtered_callback(self, gps_msg):
		self.gps_filtered = (gps_msg.latitude, gps_msg.longitude)

	def recalculate_goal(self, goal_orginal_x, goal_orginal_y):
		rospy.loginfo('Recalculating goal to include offset between gps->odom')
		rospy.loginfo('({}, {})->({}, {})'.format(self.gps_fix[0],
												  self.gps_fix[1],
												  self.gps_filtered[0],
												  self.gps_filtered[1]))
		goal_offset_x, goal_offset_y = gc.ll2xy(self.gps_filtered[0], 
												self.gps_filtered[1], 
												self.gps_fix[0],
												self.gps_fix[1])
		new_goal_x = goal_orginal_x - goal_offset_x
		new_goal_y = goal_orginal_y - goal_offset_y

		rospy.loginfo('Recalculated goal: ({}, {})'.format(new_goal_x, new_goal_y))
		return new_goal_x, new_goal_y