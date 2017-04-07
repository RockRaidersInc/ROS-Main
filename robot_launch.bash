#!/bin/bash
if [ -d "/opt/ros/kinetic" ]; then
    source /opt/ros/kinetic/setup.bash
else
    echo "WARNING: ROS Kinetic is not installed.  The rover-side code may not function properly on older versions..."
    read -p "Continue? (Y/N) " yesno
    if [ "$yesno" != "y" -a "$yesno" != "Y" ]; then
        exit 1;
    fi
    source /opt/ros/indigo/setup.bash
fi
source ../devel/setup.bash


MY_IP=$(hostname -I | grep -Eo '([0-9]*\.){3}[0-9]*')
if [ "$MY_IP" = "192.168.1.1" ]; then
    echo "Detected configuration for antenna use, if this is incorrect please check your wireless connection"
else
    echo "Detected configuration for local network use, if this is incorrect please verify that you have properly configured your IP and disabled wireless"
    echo "If this is correct, the rover IP is $MY_IP"
    echo "You will need this for the base station setup"
fi
read -p "Press [ENTER] to continue launch..."

export ROS_IP=$MY_IP
roslaunch drive startupcam.launch
