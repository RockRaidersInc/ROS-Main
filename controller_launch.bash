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

source /opt/ros/indigo/setup.bash
source ../devel/setup.bash

# Configure ROS environment
export ROS_MASTER_URI=http://"$ROVER_IP":11311
MY_IP=$(hostname -I | grep -Eo '([0-9]*\.){3}[0-9]*')
if [ "$ROVER_IP" = "192.168.1.0" -a "$MY_IP" != "192.168.1.9" ]
    then
    # Warn user if they perhaps forgot to run the antenna setup on this end
    echo "WARNING: Defualt rover IP used, but computer's IP is not default controller IP"
fi
export ROS_HOSTNAME=$MY_IP
# Launch the actual code
roslaunch controller.launch