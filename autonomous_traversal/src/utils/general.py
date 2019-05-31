#!/usr/bin/env python

import rospy
import rospkg
import yaml
from move_base_msgs.msg import MoveBaseGoal
from robot_localization.srv import SetDatum
from geographic_msgs.msg import GeoPose


def load_coordinates(filename):
	rospack = rospkg.RosPack()
	AT_file_path = rospack.get_path('autonomous_traversal')
	file_path = '{}/coordinates/{}.yaml'.format(AT_file_path, filename)
	with open(file_path, 'r') as file:
		coordinates = yaml.load(file)
	return coordinates

def load_datum_from_map_file(map_name):
	rospack = rospkg.RosPack()
	AT_file_path = rospack.get_path('autonomous_traversal')
	file_path = '{}/maps/{}.yaml'.format(AT_file_path, map_name)
	with open(file_path, 'r') as file:
		map_file = yaml.load(file)
		try:
			return map_file['datum']
		except KeyError as ke: 
			rospy.logwarn('datum not found for map file {}.yaml'.format(map_name))
			return None

def make_goal(x=0, y=0, thetaZ=1, w=0.):
	goal = MoveBaseGoal()
	goal.target_pose.header.frame_id = '/map'
	goal.target_pose.header.stamp = rospy.Time.now()
	goal.target_pose.pose.position.x = x
	goal.target_pose.pose.position.y = y
	goal.target_pose.pose.orientation.z = thetaZ
	goal.target_pose.pose.orientation.w = w
	return goal

def set_mapviz_origin(orig_lat,orig_lon):
	try:
		origin_name = rospy.get_param('local_xy_origin')
		origins = rospy.get_param('local_xy_origins')
		for origin in origins:
			if origin_name == origin['name']:
				origin['latitude'] = orig_lat
				origin['longitude'] = orig_lon
				break
		rospy.set_param('local_xy_origins', origins)
	except KeyError:
		pass

def set_datum(datum_lat, datum_lon):
	rospy.loginfo('Waiting for service datum')
	rospy.wait_for_service('datum')
	try:
		datum_srv = rospy.ServiceProxy('datum', SetDatum)
		datum_pose = GeoPose()
		datum_pose.position.latitude = datum_lat
		datum_pose.position.longitude = datum_lon
		datum_pose.position.altitude = 0
		datum_pose.orientation.x = 0
		datum_pose.orientation.y = 0
		datum_pose.orientation.z = 0
		datum_pose.orientation.w = 1
		rospy.loginfo('Setting datum to ({}, {})'.format(datum_lat, datum_lon))
		resp = datum_srv(datum_pose)
		rospy.loginfo('SetDatum Response: {}'.format(resp))
	except rospy.ServiceException as se:
		rospy.logerror('Service call failed: {}'.format(se))