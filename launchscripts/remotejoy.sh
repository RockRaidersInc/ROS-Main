#!/bin/bash
source /opt/ros/kinetic/setup.bash
#set to your workspace setup file
source ~/ws/devel/setup.bash
text1=$(curl -s http://whatismyip.akamai.com/)
echo "IP address is:"
echo $text1
export ROS_IP=$text1
echo -n "Enter target ip address: "
read text2
export ROS_MASTER_URI=http://$text2:11311
echo $ROS_MASTER_URI

roslaunch drive joystick.launch

read
