#!/usr/bin/env python

import rospy
import sys
from utils.waypoint_nav import *


def get_origin():
	while True:
		try:	
			orig_lat = float(input('Input Origin Latitude: '))
			orig_lon = float(input('Input Origin Longitude: '))
			if orig_lat < -90 or orig_lat > 90 or \
			   orig_lon <-180 or orig_lon > 180:
			   	raise Exception
			else:
				break
		except:
			rospy.loginfo('Invalid origin, try again')

	return orig_lat, orig_lon

def main():
	rospy.init_node('input_goals')
	orig_lat, orig_lon = get_origin()
	set_mapviz_origin(orig_lat, orig_lon)
	waypoint_navigator = WaypointNavigator(orig_lat, orig_lon)
	while True:
		try:	
			goal_lat = input('Input Goal Latitude: ')
			goal_lon = input('Input Goal Longitude: ')
			if goal_lat < -90 or goal_lat > 90 or \
			   goal_lon <-180 or goal_lon > 180:
			   	raise Exception

			waypoint_navigator.send_gps_goal(goal_lat, goal_lon)

			if str(raw_input('Stop (y)? ')) == 'y':
				break
		except:
			rospy.loginfo('Invalid goal, try again')

	sys.exit()


if __name__ == '__main__':
	main()