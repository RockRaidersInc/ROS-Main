#!/bin/bash

#Author: Owen Xie
#Stores the past master URI in coreip.txt and asks user if they
#want to change it. Exports both ROS_IP (from hostname -I) and ROS_MASTER_URI at the end
#
#If you will run this in your script, make sure to run it as:
#   source connection.sh

ip=$(hostname -I)
masterURI=$(cat coreip.txt) 
echo "Would you like to use the last master ip you used? (${masterURI}) [yN]" 
read input

if [ ${input} = "y" ]; then
    echo "Using ${masterURI} as the master URI!"
else
    echo "Insert a new IP:"
    read masterURI
    echo ${masterURI} > coreip.txt
fi
echo "Exporting ROS_IP as ${ip}!"
export ROS_IP=${ip}
echo "Exporting ROS_MASTER_URI as http://${masterURI}:11311!"
export ROS_MASTER_URI="http://${masterURI}:11311"

   
