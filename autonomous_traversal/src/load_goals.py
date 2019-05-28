#!/usr/bin/env python

import rospy
import sys
import time
from utils.waypoint_nav import WaypointNavigator
from utils.general import *


def main():
	rospy.init_node('load_goals')

	# Get datum from map file and set origins
	rospy.loginfo('Loading datum from map file')
	map_name = rospy.get_param('map_name')
	orig_lat, orig_lon, _ = load_datum_from_map_file(map_name)
	set_mapviz_origin(orig_lat, orig_lon)
	set_datum(orig_lat, orig_lon)

	# Load the gps goal coordinates
	coordinates_filename = rospy.get_param('coordinates')
	coordinates = load_coordinates(coordinates_filename)
	gps_goals = coordinates['gps_goals']
	
	# Start navigating
	waypoint_navigator = WaypointNavigator(orig_lat, orig_lon)
	for goal_num, (goal_lat, goal_lon) in enumerate(gps_goals):
		waypoint_navigator.send_gps_goal(goal_lat, goal_lon)
		time.sleep(1)		

	sys.exit()


if __name__ == '__main__':
	main()