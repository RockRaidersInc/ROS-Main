import rospy
import rospack
import yaml
from move_base_msgs.msg import MoveBaseGoal


def load_coordinates(filename):
	rospack = rospkg.RosPack()
	AT_file_path = rospack.get_path('autonomous_nav')
	file_path = '{}/coordinates/{}.yaml'.format(AT_file_path, filename)
	with open(file_path) as file:
		coordinates = yaml.safe_load(file)
	return coordinates

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
	origin_name = rospy.get_param('local_xy_origin')
	origins = rospy.get_param('local_xy_origins')
	for origin in origins:
		if origin_name == origin['name']:
			origin['latitude'] = orig_lat
			origin['longitude'] = orig_lon
			break
	rospy.set_param('local_xy_origins', origins)

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
