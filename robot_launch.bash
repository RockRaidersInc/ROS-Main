#!/bin/bash
source /opt/ros/indigo/setup.bash
source ../devel/setup.bash
MY_IP=$(hostname -I | grep -Eo '([0-9]*\.){3}[0-9]*')
echo $MY_IP
if [ "$MY_IP" = "192.168.1.0" ]; then
    echo "Detected configuration for antenna use, if this is incorrect please check your wireless connection"
else
    echo "Detected configuration for local network use, if this is incorrect please verify that you have properly configured your IP and disabled wireless"
fi

read -p "Press [ENTER] to continue launch..."

export ROS_IP=$MY_IP
roslaunch rover.launch