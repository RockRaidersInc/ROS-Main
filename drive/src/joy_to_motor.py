#TO DO: Add subscriber to orientation data with callback to calc_motor_limits, properly define motor limits

#!/usr/bin/env python

#Last edited by Connor McGowan 1/18/19

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8
import kinematics

class joycontrol:
    def __init__(self):
        self.left = 0
        self.right = 0

	self.motor_max = 12.7
	self.motor_min = -12.7

        #Should go in a config file
        self.min_turn_radius = 35.485/24
        self.track = 35.485/12

        rospy.init_node('joycontrol')

        self.left_pub = rospy.Publisher('left_motor_power', Int8, queue_size=1)
        self.right_pub = rospy.Publisher('right_motor_power', Int8, queue_size=1)

        rospy.Subscriber('twist_in', Twist, self.twist_callback)

        self.publish_timer = rospy.Timer(rospy.Duration(0.05), self.publish_to_motors)  # publish joystick angles every 0.05 seconds

    def publish_to_motors(self, event):

        #Publishes motor velocites IN RADIANS PER SECOND
        self.left_pub.publish(int(self.left))
        self.right_pub.publish(int(self.right))

    def calc_motor_limits(self):

        #Should calculate motor limits based on slope
        self.motor_max = self.motor_max
        self.motor_min = self.motor_min

    def twist_callback(self, data):
        
        #We can't do zero point turning, so velocity must be nonzero
        if data.linear.x==0 and data.angular.z!=0:
            data.linear.x=0.5

        #Get initial wheel angular velocities from IK
        temp_left, temp_right = kinematics.inverse_kinematics(data.linear.x, data.angular.z)

        #Don't bother doing all this if we're stopped
        if temp_right!=0 or temp_left!=0:

            #Factor used for determining minimum turn radius
            radius_limiter=(2*self.min_turn_radius-self.track)/(2*self.min_turn_radius+self.track)
            speed_factor = temp_right+temp_left
        
            #Check if radius is too tight. If so, increase speed of inside wheel to match min turn radius
            if data.linear.x>0 and data.angular.z>0 and temp_left<radius_limiter*temp_right:
                temp_left=radius_limiter*temp_right
            
            elif data.linear.x<0 and data.angular.z>0 and temp_right>radius_limiter*temp_left:
                temp_right=radius_limiter*temp_left
            elif data.linear.x>0 and data.angular.z<0 and temp_right<radius_limiter*temp_left:
                temp_right=radius_limiter*temp_left
            elif data.linear.x<0 and data.angular.z<0 and temp_left>radius_limiter*temp_right:
                temp_left=radius_limiter*temp_right

            #Scale velocities back down so linear velocity is the same
            new_speed_factor=temp_left+temp_right
            temp_right*=speed_factor/new_speed_factor
            temp_right*=speed_factor/new_speed_factor

            #Check if motor limits are exceeded. If so, clip to limit and reduce other motor proportionally
            if temp_left > self.motor_max:
                temp_right *= self.motor_max/temp_left
                temp_left = self.motor_max
            elif temp_left < self.motor_min:
                temp_right *= self.motor_min/temp_left
                temp_left = self.motor_min
            if temp_right > self.motor_max:
                temp_left *= self.motor_max/temp_right
                temp_right = self.motor_max
            elif temp_right < self.motor_min:
                temp_left *= self.motor_min/temp_right
                temp_right = self.motor_min

        #Set final angular velocities 
        self.left = temp_left
        self.right = temp_right

if __name__ == '__main__':
    joy = joycontrol()
    rospy.spin()
