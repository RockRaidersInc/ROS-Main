#!/usr/bin/env python


import rospy
import actionlib
import sys
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
import geonav_transform.geonav_conversions as gc


def make_goal(x=0, y=0, thetaZ=1., w=0.):
	goal = MoveBaseGoal()
	goal.target_pose.header.frame_id = 'map'
	goal.target_pose.header.stamp = rospy.Time.now()
	goal.target_pose.pose.position.x = x
	goal.target_pose.pose.position.y = y
	goal.target_pose.pose.orientation.z = thetaZ
	goal.target_pose.pose.orientation.w = w
	return goal

def main2():
	rospy.init_node('send_goal')
	client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
	rospy.loginfo('Waiting for server')
	client.wait_for_server()

	olat=42.729957
	olon=-73.679659
	target_lat = 42.7301
	target_lon = -73.679659
	xg2,yg2 = gc.ll2xy(target_lat,target_lon,olat,olon)
	checkpoint = (xg2,yg2)
	rospy.loginfo('Making goal with (lat,lon): ({},{})'.format(target_lat,target_lon))
	rospy.loginfo('(lat,lon): ({},{}) converts to (x,y): ({},{})'.format(target_lat,target_lon,xg2,yg2))

	goal = make_goal(*checkpoint)
	rospy.loginfo('Sending goal')
	client.send_goal(goal)
	rospy.loginfo('Waiting for results')
	client.wait_for_result()
	sys.exit()


def main():
	rospy.init_node('send_goal')
	client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
	rospy.loginfo('Waiting for server')
	client.wait_for_server()
	checkpoint = (0,0)
	rospy.loginfo('Making goal with checkpoint: {}'.format(checkpoint))
	goal = make_goal(*checkpoint)
	rospy.loginfo('Sending goal')
	client.send_goal(goal)
	rospy.loginfo('Waiting for results')
	client.wait_for_result()
	sys.exit()


if __name__ == '__main__':
	main2()