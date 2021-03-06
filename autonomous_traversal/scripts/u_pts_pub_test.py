import rospy
from std_msgs.msg import Float64MultiArray

def u_pts_pub_test():
	pub = rospy.Publisher('/anti_rev_u_pts', Float64MultiArray, queue_size=10)
	rospy.init_node('u_pts_pub_test')

	while not rospy.is_shutdown():
		keystroke = raw_input()

		u_pts = Float64MultiArray()
		if keystroke == 'r':
			print('Clearing u obstacles')
			u_pts.data = [0,0,0,0,0,0,0,0]
		else:
			print('Adding u obstacle')
			u_pts.data = [2.5,1.5,-1.5,1.5,-1.5,-1.5,2.5,-1.5]
		
		pub.publish(u_pts)

if __name__ == '__main__':
	try:
		u_pts_pub_test()
	except rospy.ROSInterruptException:
		pass