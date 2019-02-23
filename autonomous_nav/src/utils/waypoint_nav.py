#!/usr/bin/env python

import rospy
import rospkg
import actionlib
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
import geonav_transform.geonav_conversions as gc


def set_mapviz_origin(orig_lat,orig_lon):
	origin_name = rospy.get_param('local_xy_origin')
	origins = rospy.get_param('local_xy_origins')
	for origin in origins:
		if origin_name == origin['name']:
			origin['latitude'] = orig_lat
			origin['longitude'] = orig_lon
			break
	rospy.set_param('local_xy_origins', origins)

def make_goal(x=0, y=0, thetaZ=1, w=0.):
	goal = MoveBaseGoal()
	goal.target_pose.header.frame_id = '/map'
	goal.target_pose.header.stamp = rospy.Time.now()
	goal.target_pose.pose.position.x = x
	goal.target_pose.pose.position.y = y
	goal.target_pose.pose.orientation.z = thetaZ
	goal.target_pose.pose.orientation.w = w
	return goal

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