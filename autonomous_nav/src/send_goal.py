#!/usr/bin/env python


import rospy
import actionlib
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction


def make_goal(x=0, y=0, thetaZ=1., w=0.):
	goal = MoveBaseGoal()
	goal.target_pose.header.frame_id = 'odom'
	goal.target_pose.header.stamp = rospy.Time.now()
	goal.target_pose.pose.position.x = x
	goal.target_pose.pose.position.y = y
	goal.target_pose.pose.orientation.z = thetaZ
	goal.target_pose.pose.orientation.w = w
	return goal

def main():
	rospy.init_node('send_goal')
	client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
	rospy.loginfo('Waiting for server')
	client.wait_for_server()
	checkpoint = (10,0)
	rospy.loginfo('Making goal with checkpoint: {}'.format(checkpoint))
	goal = make_goal(10, 0)
	rospy.loginfo('Sending goal')
	client.send_goal(goal)
	rospy.loginfo('Waiting for results')
	client.wait_for_result()
	sys.exit()


if __name__ == '__main__':
	main()