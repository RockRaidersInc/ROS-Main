#!/usr/bin/env python

import rospy
import sys
import time
from sensor_msgs.msg import NavSatFix
from utils.waypoint_nav import WaypointNavigator
from utils.general import *


# TODO: Put all this stuff in WaypointNavigator 
current_gps_loc = None

def save_gps_loc_callback(gps_msg):
	global current_gps_loc
	current_gps_loc = (gps_msg.latitude, gps_msg.longitude)

def input_origin():
	while True:
		# Current location as origin
		if str(raw_input('Grab current GPS coordinate as origin (y): ')) == 'y':
			if current_gps_loc is None:
				rospy.logwarn('No current GPS coordinate to grab')
				continue
			else:
				orig_lat, orig_lon = current_gps_loc
				rospy.loginfo('origin ({}, {})'.format(*current_gps_loc))
		# Origin from map file 
		elif str(raw_input('Grab datum from map file as origin (y): ')) == 'y':
			try:
				map_name = rospy.get_param('map_name')
			except KeyError as ke:
				rospy.logwarn('map_name parameter not set, can\'t find map file')
				continue
			map_file_datum = load_datum_from_map_file(map_name)
			if map_file_datum is not None:
				orig_lat, orig_lon, _ = map_file_datum
			else:
				continue
		# Manually input the GPS origin
		else:
			orig_lat = float(raw_input('Input Origin Latitude: '))
			orig_lon = float(raw_input('Input Origin Longitude: '))
			
		# Bound check GPS coordinate		
		if orig_lat < -90 or orig_lat > 90 or \
		   orig_lon <-180 or orig_lon > 180:
			rospy.logwarn('Invalid origin ({}, {}), try again'.format(orig_lat, orig_lon))
		else:
			break

	return orig_lat, orig_lon
# TODO: end


def main():
	rospy.init_node('load_goals')
	rospy.Subscriber('gps/fix', NavSatFix, save_gps_loc_callback)
	orig_lat, orig_lon = input_origin()
	set_mapviz_origin(orig_lat, orig_lon)
	set_datum(orig_lat, orig_lon)

	# Load the gps goal coordinates
	coordinates_filename = rospy.get_param('coordinates')
	coordinates = load_coordinates(coordinates_filename)
	gps_goals = coordinates['gps_goals']
	
	# Check if we want to dynamically recalculate goals
	recalc_goals = rospy.get_param('recalc_goals')
	if recalc_goals:
		waypoint_navigator = WaypointNavigator(orig_lat, 
											   orig_lon, 
											   recalc_goals=True, 
											   recalc_rate=3.0)
	else:
		waypoint_navigator = WaypointNavigator(orig_lat, orig_lon)	
	# Start navigating
	for goal_num, (goal_lat, goal_lon) in enumerate(gps_goals):
		waypoint_navigator.send_gps_goal(goal_lat, goal_lon)
		time.sleep(1)		

	sys.exit()


if __name__ == '__main__':
	main()