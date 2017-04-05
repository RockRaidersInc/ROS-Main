#!/bin/bash
# Defualt rover IP when used with antenna
ROVER_IP=192.168.1.0

if [ $# -eq 1 ] # User overrode default rover IP
    then
    ROVER_IP=$1
    echo "Rover IP set to "$ROVER_IP
else
    echo "No rover IP set, using default (192.168.1.0)"
fi

if [ -d "/opt/ros/kinetic" ]; then
    source /opt/ros/kinetic/setup.bash
else
    source /opt/ros/indigo/setup.bash
fi
source ../devel/setup.bash

# Configure ROS environment
export ROS_MASTER_URI=http://"$ROVER_IP":11311
MY_IP=$(hostname -I | grep -Eo '([0-9]*\.){3}[0-9]*')
if [ "$ROVER_IP" = "192.168.1.0" -a "$MY_IP" != "192.168.1.9" ]
    then
    # Warn user if they perhaps forgot to run the antenna setup on this end
    echo "WARNING: Defualt rover IP used, but computer's IP is not default controller IP"
    read -p "Do you want to continue launch? (Y/N) " yesno 
    if [ "$yesno" != "y" -a "$yesno" != "Y" ]; then
        exit 1;
    fi
fi
export ROS_IP=$MY_IP
# Launch the actual code
roslaunch drive joycam.launch
