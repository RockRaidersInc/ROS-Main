#!/usr/bin/env python

import rospy
import sys
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from utils.waypoint_nav import WaypointNavigator
from utils.general import set_mapviz_origin
from geonav_transform import geonav_conversions as gc


current_gps_loc = None


def save_gps_loc_callback(gps_msg):
	global current_gps_loc
	current_gps_loc = (gps_msg.latitude, gps_msg.longitude)

def input_origin():
	while True:
		try:	
			# Grab current GPS origin (yes or no)
			grab_current = str(raw_input('Grab current GPS coordinate as origin (y): '))
			if grab_current == 'y':
				if current_gps_loc is None:
					rospy.logwarn('No current GPS coordinate to grab')
					continue
				else:
					orig_lat, orig_lon = current_gps_loc
					rospy.loginfo('origin ({}, {})'.format(*current_gps_loc))
			else:
				orig_lat = float(raw_input('Input Origin Latitude: '))
				orig_lon = float(raw_input('Input Origin Longitude: '))

			# Bound check GPS coordinate		
			if orig_lat < -90 or orig_lat > 90 or \
			   orig_lon <-180 or orig_lon > 180:
				rospy.logwarn('Invalid origin ({}, {}), try again'.format(orig_lat, orig_lon))
			else:
				break
		except:
			rospy.logwarn('Invalid input, try again')

	return orig_lat, orig_lon

def set_map_server_origin(start_lat, start_lon):
	map_name = rospy.get_param('/map_server/map_name')
	map_origin_lat, map_origin_lon, _ = rospy.get_param('/map_server/gps_map_origins/{}'.format(map_name))
	ekf_start_x, ekf_start_y = gc.ll2xy(start_lat, start_lon, map_origin_lat, map_origin_lon)
	print('(map_origin_lat, map_origin_lon): ({}, {})'.format(map_origin_lat, map_origin_lon))
	print('(ekf_start_x, ekf_start_y): ({}, {})'.format(ekf_start_x, ekf_start_y))
	
	

def main():
	rospy.init_node('input_goals')
	rospy.Subscriber('gps/filtered', NavSatFix, save_gps_loc_callback)


	orig_lat, orig_lon = input_origin()
	set_mapviz_origin(orig_lat, orig_lon)
	set_map_server_origin(orig_lat, orig_lon)
	waypoint_navigator = WaypointNavigator(orig_lat, orig_lon)

	while not rospy.is_shutdown():
		try:	
			# Input GPS coordinate
			goal_lat = float(raw_input('Input Goal Latitude: '))
			goal_lon = float(raw_input('Input Goal Longitude: '))

			# Bound check goal before sending it to the navigator
			if goal_lat < -90 or goal_lat > 90 or \
			   goal_lon <-180 or goal_lon > 180:
				rospy.logwarn('Invalid goal ({}, {}), try again'.format(goal_lat, goal_lon))
				continue
			else:
				rospy.loginfo('Sending goal ({}, {})'.format(goal_lat, goal_lon))
				waypoint_navigator.send_gps_goal(goal_lat, goal_lon)

			# Stop?
			if str(raw_input('Stop (y)? ')) == 'y':
				break
		except:
			rospy.logwarn('Invalid input, try again')

	sys.exit()

if __name__ == '__main__':
	main()