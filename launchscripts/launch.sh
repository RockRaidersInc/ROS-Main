#!/bin/bash
source /opt/ros/kinetic/setup.bash
source ~/ws/devel/setup.bash

export CATKIN_WS_ROOT_DIR="$(dirname $(readlink -f $0))"

text1=$(curl -s http://whatismyip.akamai.com/)

echo "IP address is:"
echo $text1
echo 
echo
roslaunch launch_files start_rover.launch
