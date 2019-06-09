#!/usr/bin/env python

import rospy
import sys
import time
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64MultiArray
from utils.waypoint_nav import WaypointNavigator
from utils.general import *


# TODO: Put all this stuff in WaypointNavigator 
current_gps_loc = None
U_OBST_CLEAR = [0,0,0,0,0,0,0,0]

def save_gps_loc_callback(gps_msg):
	global current_gps_loc
	current_gps_loc = (gps_msg.latitude, gps_msg.longitude)

def grab_origin():
	while True:
		# Current location as origin
		if current_gps_loc is None:
			rospy.logwarn('No current GPS coordinate to grab')
			time.sleep(1)
			continue
		else:
			orig_lat, orig_lon = current_gps_loc
			rospy.loginfo('origin ({}, {})'.format(orig_lat, orig_lon))			
		# Bound check GPS coordinate		
		if orig_lat < -90 or orig_lat > 90 or \
		   orig_lon <-180 or orig_lon > 180:
			rospy.logwarn('Invalid origin ({}, {}), try again'.format(orig_lat, orig_lon))
		else:
			break
	return orig_lat, orig_lon

# TODO
def wait_for_estop_signal():
	pass

# TODO
def blink_lights():
	pass

def publish_U_obst(pub, pts):
	if pts.count(0) == 8:
		rospy.loginfo('Clearing previous U obst')
	else:
		rospy.loginfo('Publishing U obst with corners: \
					   ({}, {}), ({}, {}), ({}, {}), ({}, {})'.format(*pts))
	u_pts = Float64MultiArray()
	u_pts.data = pts
	pub.publish(u_pts)

def main():
	# Init ros stuff
	rospy.init_node('igvc')
	rospy.Subscriber('gps/fix', NavSatFix, save_gps_loc_callback)
	u_obsts_pub = rospy.Publisher('/anti_rev_u_pts', Float64MultiArray, queue_size=10)
	# Grab origin GPS loc and load the gps goal coordinates
	orig_lat, orig_lon = grab_origin()
	coordinates_filename = rospy.get_param('coordinates')
	coordinates = load_coordinates(coordinates_filename)
	gps_goals = coordinates['gps_goals']
	u_obsts = coordinates['u_obsts']
	# TODO: Wait for signal from wireless switch
	wait_for_estop_signal()
	# Check if we want to dynamically recalculate goals
	recalc_goals = rospy.get_param('recalc_goals')
	if recalc_goals:
		waypoint_navigator = WaypointNavigator(orig_lat, 
											   orig_lon, 
											   recalc_goals=recalc_goals, 
											   recalc_rate=3.0)
	else:
		waypoint_navigator = WaypointNavigator(orig_lat, orig_lon)	
	# Start navigating
	for gps_goal, u_obst in zip(gps_goals, u_obsts):
		publish_U_obst(u_obsts_pub, u_obst)
		time.sleep(1)
		# Send GPS goal
		goal_lat, goal_lon = gps_goal
		status = waypoint_navigator.send_gps_goal(goal_lat, goal_lon)
		time.sleep(1)
		# Clear the u obstacle
		#### THIS IS NOT WORKING RIGHT NOW, I NEED TO FIX THIS
		# publish_U_obst(u_obsts_pub, U_OBST_CLEAR)
		# time.sleep(1)
		# TODO: Signal the lights
		blink_lights()

	sys.exit()


if __name__ == '__main__':
	main()