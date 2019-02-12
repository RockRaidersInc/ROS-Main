#!/bin/bash

# for printing in color
green='\e[0;32m'
red='\e[0;31m'
endColor='\e[0m'

if [[ $- == *i* ]]
then
    export ROS_HOSTNAME=$(hostname).local
    export ROS_MASTER_URI=http://rock-desktop.local:11311/

    echo ROS_HOSTNAME set to $ROS_HOSTNAME
    echo ROS_MASTER_URI set to $ROS_MASTER_URI
    echo -e ${green}now run \"roslaunch launch_files base_station.launch\" to launch the joystick node on the local machine${endColor}
else
    echo -e ${red}this script must be run as \"source export_remote_ros_vars.sh\", not \"./export_remote_ros_vars.sh\"${endColor}
    exit 1
fi
