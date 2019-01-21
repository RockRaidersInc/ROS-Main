#!/bin/bash


SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH
cd $SCRIPTPATH

source /opt/ros/kinetic/setup.bash
source ../../devel/setup.bash

#source /opt/ros/kinetic/setup.bash
#source ~/ws/devel/setup.bash

export CATKIN_WS_ROOT_DIR="$(dirname $(readlink -f $0))"

text1=$(curl -s http://whatismyip.akamai.com/)

echo "IP address is:"
echo $text1
echo 
echo
roslaunch launch_files start_rover.launch
