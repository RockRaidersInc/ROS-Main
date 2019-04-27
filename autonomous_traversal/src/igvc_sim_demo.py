#!/usr/bin/env python

import rospy
import sys
import time
import actionlib
from move_base_msgs.msg import MoveBaseAction
from std_msgs.msg import Float64MultiArray
from utils.general import load_coordinates
from utils.general import make_goal


U_OBST_PTS = [2.5,1.5,-1.5,1.5,-1.5,-1.5,2.5,-1.5]
U_OBST_CLEAR = [0,0,0,0,0,0,0,0]


def publish_U_obst(pub, pts):
	if sum(pts) == 0:
		rospy.loginfo('Clearing previous U obst')
	else:
		rospy.loginfo('Publishing U obst with corners: \
					   ({}, {}), ({}, {}), ({}, {}), ({}, {})'.format(*pts))
	u_pts = Float64MultiArray()
	u_pts.data = pts
	pub.publish(u_pts)

def send_goal(mb_client, waypt_x, waypt_y):
	mb_client.wait_for_server()
	rospy.loginfo('------------------------------------------')
	rospy.loginfo('Making goal with (x,y): ({},{})'.format(waypt_x, waypt_y))
	move_base_goal = make_goal(waypt_x, waypt_y)
	rospy.loginfo('Sending goal')
	mb_client.send_goal(move_base_goal)
	rospy.loginfo('Waiting for results')
	mb_client.wait_for_result()

def main():
	# ROS stuff
	u_obst_pub = rospy.Publisher('/anti_rev_u_pts', Float64MultiArray, queue_size=10)
	rospy.init_node('igvc_sim_demo')

	# Load coordinates 
	coordinates_filename = 'igvc_sim_demo'
	rospy.loginfo('Loading waypoints from file {}'.format(coordinates_filename))
	coordinates = load_coordinates(coordinates_filename)
	waypts = coordinates['waypoints']

	# Create move_base client
	rospy.loginfo('Creating move_base client')
	mb_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
	rospy.loginfo('Waiting for server')
	mb_client.wait_for_server()

	for waypt_num, (waypt_x, waypt_y) in enumerate(waypts):
		# Publish U shaped obstacle to prevent going backwards on startup
		publish_U_obst(u_obst_pub, U_OBST_PTS)
		time.sleep(1)
		# Make and send goal to move_base
		send_goal(mb_client, waypt_x, waypt_y)
		# Clear U shaped obstacle that was previously published
		publish_U_obst(u_obst_pub, U_OBST_CLEAR)
		time.sleep(1)

	sys.exit()


if __name__ == '__main__':
	main()