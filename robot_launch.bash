#!/bin/bash
source /opt/ros/indigo/setup.bash
source ../devel/setup.bash
export ROS_HOSTNAME=$(hostname -I | grep -Eo '([0-9]*\.){3}[0-9]*')
roslaunch rover.launch