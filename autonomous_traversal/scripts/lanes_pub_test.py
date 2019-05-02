import rospy
from autonomous_traversal.msg import Lane
from geometry_msgs.msg import Vector3

def u_pts_pub_test():
	pub = rospy.Publisher('/lanes', Lane, queue_size=10)
	rospy.init_node('lanes_pub_test')
	rate = rospy.Rate(30)

	while not rospy.is_shutdown():
		keystroke = raw_input()

		lane = Lane()
		for i in range(10):
			lane_pt = Vector3(i,i,0)	
			lane.lane_points.append(lane_pt)
		pub.publish(lane)

		# rate.sleep()

if __name__ == '__main__':
	try:
		u_pts_pub_test()
	except rospy.ROSInterruptException:
		pass