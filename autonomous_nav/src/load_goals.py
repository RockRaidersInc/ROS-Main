#!/usr/bin/env python

import rospy
import sys
import time
from utils.waypoint_nav import *


def main():
	rospy.init_node('load_goals')

	# coordinates_filename = rospy.get_param('coordinates')
	coordinates_filename = '86_field'
	coordinates = load_coordinates(coordinates_filename)
	orig_lat, orig_lon = coordinates['origin']
	gps_goals = coordinates['gps_goals']

	waypoint_navigator = WaypointNavigator(orig_lat, orig_lon)
	for goal_num, (goal_lat, goal_lon) in enumerate(gps_goals):
		waypoint_navigator.send_gps_goal(goal_lat, goal_lon)
		time.sleep(1)		

	sys.exit()


if __name__ == '__main__':
	main()