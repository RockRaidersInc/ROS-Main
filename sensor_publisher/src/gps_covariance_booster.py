#!/usr/bin/env python

import rospy
from sensor_msgs.msg import NavSatFix


def callback(msg):
    msg.position_covariance = [val * adjustment_factor for val in msg.position_covariance]
    pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("gps_covariance_booster")
    adjustment_factor = rospy.get_param('~gps_scale_factor')
    rospy.loginfo('gps_scale_factor: {}'.format(adjustment_factor))
    sub = rospy.Subscriber("~gps_msg_in", NavSatFix, callback, queue_size = 2)
    pub = rospy.Publisher("~gps_msg_out", NavSatFix, queue_size = 2)
    rospy.spin()
