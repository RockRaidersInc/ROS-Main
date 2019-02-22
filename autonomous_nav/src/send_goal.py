#!/usr/bin/env python


import rospy
import rospkg
import actionlib
import sys
import yaml
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
import geonav_transform.geonav_conversions as gc


def make_goal(x=0, y=0, thetaZ=1, w=0.):
	goal = MoveBaseGoal()
	goal.target_pose.header.frame_id = '/map'
	goal.target_pose.header.stamp = rospy.Time.now()
	goal.target_pose.pose.position.x = x
	goal.target_pose.pose.position.y = y
	goal.target_pose.pose.orientation.z = thetaZ
	goal.target_pose.pose.orientation.w = w
	return goal

def load_coordinates(filename):
	rospack = rospkg.RosPack()
	AT_file_path = rospack.get_path('autonomous_nav')
	file_path = '{}/coordinates/{}.yaml'.format(AT_file_path, filename)
	with open(file_path) as file:
		coordinates = yaml.safe_load(file)
	return coordinates


class WaypointNavigator:
	def __init__(self, orig_lat, orig_lon):
		rospy.init_node('waypoint_navigator')
		self.orig_lat = orig_lat
		self.orig_lon = orig_lon
		rospy.loginfo('Creating move_base client')
		self.mb_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
		rospy.loginfo('Waiting for server')
		self.mb_client.wait_for_server()

	def send_gps_goal(self, goal_lat, goal_lon):
		self.mb_client.wait_for_server()
		rospy.loginfo('----------Making goal with (lat,lon): ({},{})----------'\
					  .format(goal_lat, goal_lon))
		goal_x, goal_y = gc.ll2xy(goal_lat, goal_lon,
								  self.orig_lat, self.orig_lon)
		rospy.loginfo('(lat,lon): ({},{}) converts to \n(x,y): ({},{})'\
					  .format(goal_lat, goal_lon, goal_x, goal_y))
		move_base_goal = make_goal(goal_x, goal_y)
		rospy.loginfo('Sending goal')
		self.mb_client.send_goal(move_base_goal)
		rospy.loginfo('Waiting for results')
		self.mb_client.wait_for_result()


def main():
	# coordinates_filename = rospy.get_param('coordinates')
	coordinates_filename = '86_field'
	coordinates = load_coordinates(coordinates_filename)
	orig_lat, orig_lon = coordinates['origin']
	gps_goals = coordinates['gps_goals']
	waypoint_navigator = WaypointNavigator(orig_lat, orig_lon)
	for goal_num, (goal_lat, goal_lon) in enumerate(gps_goals):
		waypoint_navigator.send_gps_goal(goal_lat, goal_lon)	


if __name__ == '__main__':
	main()