#!/bin/bash


green='\e[0;32m'
red='\e[0;31m'
endColor='\e[0m'

echo -e ${red}"this file is depreicated, strongly consider starting start_with_hardware.launch and base_station.launch instead (on different machines if desired)."${endcolor}


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
roslaunch launch_files start_with_hardware.launch
