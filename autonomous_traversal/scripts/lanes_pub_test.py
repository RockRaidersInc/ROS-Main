import rospy
import time
from autonomous_traversal.msg import Lane
from geometry_msgs.msg import Vector3


def u_pts_pub_test():
	pub = rospy.Publisher('/lanes', Lane, queue_size=10)
	rospy.init_node('lanes_pub_test')
	rate = rospy.Rate(30)
	print("alive")

	while not rospy.is_shutdown():
		keystroke = raw_input()
		print('Publishing lane pts')
		
		lane = Lane()
		
		lane.bound_polygon.append(Vector3(1.1,1.1,0.0))
		lane.bound_polygon.append(Vector3(1.1,-1.1,0.0))
		lane.bound_polygon.append(Vector3(-1.1,-1.1,0.0))
		lane.bound_polygon.append(Vector3(-1.1,1.1,0.0))

		for i in range(10):
			lane_pt = Vector3(i/10.0,1.0,0.0)	
			lane.lane_points.append(lane_pt)
		for i in range(10):
			lane_pt = Vector3(i/10.0,-1.0,0.0)	
			lane.lane_points.append(lane_pt)

		# time.sleep(1)
		pub.publish(lane)

		# rate.sleep()

if __name__ == '__main__':
	try:
		u_pts_pub_test()
	except rospy.ROSInterruptException:
		pass