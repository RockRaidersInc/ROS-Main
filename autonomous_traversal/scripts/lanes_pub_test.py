import rospy
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
		
		tl_bound_pt = Vector3(1.1,1.1,0.0)
		br_bound_pt = Vector3(-.1,-1.1,0.0)
		lane.bound_corners[0] = tl_bound_pt
		lane.bound_corners[1] = br_bound_pt

		for i in range(10):
			lane_pt = Vector3(i/10.0,1.0,0.0)	
			lane.lane_points.append(lane_pt)
		# for i in range(10):
		# 	lane_pt = Vector3(i/10.0,-1.0,0.0)	
		# 	lane.lane_points.append(lane_pt)

		pub.publish(lane)

		# rate.sleep()

if __name__ == '__main__':
	try:
		u_pts_pub_test()
	except rospy.ROSInterruptException:
		pass