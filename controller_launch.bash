#!/bin/bash
if [ $# -eq 1 ]
    then
    source /opt/ros/indigo/setup.bash
    source ../devel/setup.bash
    echo export ROS_MASTER_URI=http://$1:11311
    echo export ROS_HOSTNAME=$(hostname -I | grep -Eo '([0-9]*\.){3}[0-9]*')
    roslaunch controller.launch
else
    echo No robot IP address specified
fi